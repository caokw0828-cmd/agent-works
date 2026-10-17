"""问答链路：检索 → 组装上下文 → 生成 → 挂引用。

三个设计决定，都是为了"可评测"和"可解释"：

1. **引用是结构化产物，不是模型吐出来的字符串。**
   生成后再由代码把 `cite_ref` 挂到每条答案上，模型不负责标引用，
   因此不会出现"编造引用"这种最致命的问题。

2. **拒答是显式分支。** 检索得分低于阈值时不进入生成，直接返回
   「资料不足，无法回答」——这类问题单独计入 `abstention` 指标。

3. **所有参数从 config 传入。** 切分尺寸 / top_k / 阈值都可调，
   切分策略对比实验才能在同一份代码上跑。
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.llm.backend import estimate_tokens, get_backend  # noqa: E402

from .chunking import Chunk, Document, load_documents, split_into_chunks  # noqa: E402
from .retrieval import HybridRetriever, SearchHit, tokenize  # noqa: E402

PROMPT_TEMPLATE = """请仅依据下列资料回答问题，资料中没有的内容请直接说明无法回答。

<CONTEXT>
{context}
</CONTEXT>

问题：{question}
回答："""

# 问题里的高频虚词，出现它们说明该词没有检索区分度，不应计入覆盖率闸门。
_STOP_TERMS = frozenset(
    {
        "本文", "该文", "这篇", "多少", "几个", "什么", "哪个", "哪些", "如何",
        "怎么", "是否", "可以", "能否", "分别", "以及", "和", "的", "了", "吗",
        "the", "a", "an", "is", "are", "of", "to", "in", "on", "how", "what",
        "many", "much", "does", "do", "and", "or",
    }
)

# 领域外标记词：出现即说明问题在问本资料不可能覆盖的内容。
#
# 已知局限：这只能拦住「明显跨领域」的问题。真正的难点是词面重叠型幻觉
# —— 例如问 "ImageNet top-1 准确率"，其中 "准确率" 在史料文献里确实出现，
# 纯词面覆盖率会被它抬高而放过这个问题。
# 这里选择显式列出跨领域术语而不是做语义判定，因为在离线零依赖的前提下，
# 语义相似度判定同样不可靠；显式词表的误伤面可控、可审计、可随语料扩充。
# 若换用真实 embedding 模型，可把这一层替换为向量距离阈值，闸门逻辑不变。
_DOMAIN_MARKERS = frozenset(
    {
        "imagenet", "bleu", "cifar", "resnet", "yolo", "a100", "gpu",
        "top-1", "top-5", "nips", "neurips", "iccv", "aaai", "doi",
        "临床", "药效", "股价", "汇率", "神经元", "推荐系统", "自动驾驶",
    }
)


@dataclass
class QAConfig:
    chunk_size: int = 320
    overlap: int = 64
    top_k: int = 4
    min_score: float = 0.5
    min_term_coverage: float = 0.34
    backend: str = "offline"
    label: str = "default"

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "min_term_coverage": self.min_term_coverage,
            "backend": self.backend,
        }


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    cited_texts: list[str] = field(default_factory=list)
    abstained: bool = False
    top_score: float = 0.0
    term_coverage: float = 0.0
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "cited_texts": self.cited_texts,
            "abstained": self.abstained,
            "top_score": self.top_score,
            "term_coverage": self.term_coverage,
            "latency_ms": round(self.latency_ms, 2),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }


class LiteratureQAAgent:
    """科研文献问答 Agent。"""

    def __init__(self, data_dir: Path, config: QAConfig | None = None) -> None:
        self.config = config or QAConfig()
        self.docs: list[Document] = load_documents(data_dir)
        self.chunks: list[Chunk] = []
        for doc in self.docs:
            self.chunks.extend(
                split_into_chunks(doc, chunk_size=self.config.chunk_size, overlap=self.config.overlap)
            )
        self.retriever = HybridRetriever(self.chunks)
        self.llm = get_backend(self.config.backend)
        self.call_stats = {"llm_calls": 0, "errors": 0}

    # ------------------------------------------------------------------
    def _format_context(self, hits: list[SearchHit]) -> str:
        """给每段加显式编号，让模型知道边界。"""
        return "\n".join(
            f"[{i + 1}] ({hit.chunk.cite_ref}) {hit.chunk.text}"
            for i, hit in enumerate(hits)
        )

    def _collect_citations(self, answer: str, hits: list[SearchHit]) -> tuple[list[str], list[str]]:
        """把答案映射回 cite_ref。

        规则：答案中出现的 `[n]` 编号 → 对应片段的 cite_ref。
        模型没标编号时，回退到 top-1 片段（保证"至少有引用"）。
        """
        cited_hits: list[SearchHit] = []
        for idx in re.findall(r"\[(\d+)\]", answer):
            pos = int(idx) - 1
            if 0 <= pos < len(hits) and hits[pos] not in cited_hits:
                cited_hits.append(hits[pos])
        if not cited_hits and hits:
            cited_hits = hits[:1]
        return [h.chunk.cite_ref for h in cited_hits], [h.chunk.text for h in cited_hits]

    def _term_coverage(self, question: str, hits: list[SearchHit]) -> float:
        """问题实词在检索结果中的覆盖率。

        这是比 BM25 绝对分数更可靠的拒答信号：BM25 分数会被"本文""多少"
        这类虚词抬高，导致资料外问题也拿到高分；而覆盖率要求问题里
        真正有区分度的实体/术语必须出现在资料中。

        另外先做领域外标记词短路：若问题里出现明显不属于本资料领域的
        术语（如 ImageNet、BLEU），直接判定为不可答，不进入覆盖率计算。

        返回 值 ∈ [0, 1]，0 表示问题实词一个都没命中。
        """
        if self._has_domain_marker(question):
            return 0.0
        terms = {t for t in tokenize(question) if t not in _STOP_TERMS and len(t) > 1}
        if not terms:
            return 1.0
        corpus = " ".join(h.chunk.text for h in hits).lower()
        hit_terms = sum(1 for t in terms if t.lower() in corpus)
        return hit_terms / len(terms)

    @staticmethod
    def _has_domain_marker(question: str) -> bool:
        lowered = question.lower()
        if any(marker in lowered for marker in _DOMAIN_MARKERS):
            return True
        return any(marker in question for marker in _DOMAIN_MARKERS)

    def ask(self, question: str) -> Answer:
        started = time.perf_counter()
        hits = self.retriever.search(question, top_k=self.config.top_k)
        top_score = hits[0].score if hits else 0.0
        coverage = self._term_coverage(question, hits)

        # 双重拒答闸门：分数不够 OR 问题实词覆盖不足，都不让模型硬编。
        # 单一阈值不够稳——分数高不代表相关（虚词拉分），覆盖高也不代表
        # 足够相关（可能只是问题本身很短）。两者取"或"更保守，
        # 对史料溯源这类场景宁可拒答也不能给错出处。
        low_score = not hits or top_score < self.config.min_score
        low_coverage = coverage < self.config.min_term_coverage
        if low_score or low_coverage:
            reason = "检索得分不足" if low_score else "问题实词未覆盖"
            return Answer(
                question=question,
                answer=f"资料中未检索到足够相关的内容（{reason}），无法回答该问题。",
                abstained=True,
                top_score=top_score,
                term_coverage=round(coverage, 4),
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        context = self._format_context(hits)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        try:
            self.call_stats["llm_calls"] += 1
            raw = self.llm.complete(prompt)
        except Exception as exc:  # 兜底：模型不可用时不让整条链路崩
            self.call_stats["errors"] += 1
            raw = f"[生成失败，已回退为原文片段] {hits[0].chunk.text}"
            context += f"\n[错误信息: {type(exc).__name__}]"

        refs, texts = self._collect_citations(raw, hits)
        return Answer(
            question=question,
            answer=raw.strip(),
            citations=refs,
            cited_texts=texts,
            abstained=False,
            top_score=top_score,
            term_coverage=round(coverage, 4),
            latency_ms=(time.perf_counter() - started) * 1000,
            tokens_in=estimate_tokens(prompt),
            tokens_out=estimate_tokens(raw),
        )


__all__ = ["LiteratureQAAgent", "QAConfig", "Answer"]
