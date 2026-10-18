"""LangChain 版切分与向量检索。

项目一有两套检索实现，跑同一份评测集：

- `bm25`   纯 Python 的 BM25 + 短语覆盖，不依赖 faiss，作为对照组
- `vector` LangChain 的 RecursiveCharacterTextSplitter切分 + FAISS 向量检索

两套共用同一个 `Chunk` 结构，因此**引用锚点的口径完全一致**，
`doc_id#p1#c12` 在两种检索下指向同一段原文。这样对比实验才有意义 ——
差异只来自检索方式，不来自切分或引用逻辑。

为什么两种都留着：BM25 更容易暴露切分策略的缺陷，做切分实验更干净；
向量检索则是真实 RAG 的样子。看过两者的差距，才知道什么时候该用哪个。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .chunking import Chunk

_CJK = re.compile(r"[一-鿿]")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*|\d+(?:\.\d+)?")


def _langchain_bases():
    """取 LangChain 的 Embeddings 基类；没装时退化成 object。

    这样 `bm25` 对照组在只装 LangChain、不装 faiss 的环境下仍可运行。
    """
    try:
        from langchain_core.embeddings import Embeddings
        return Embeddings, True
    except Exception:  # pragma: no cover
        return object, False


_EmbeddingsBase, HAS_LANGCHAIN = _langchain_bases()


class _LazyEmbeddingsBase(_EmbeddingsBase):  # type: ignore[misc,valid-type]
    """只有真正被 `vector` 检索用到时才要求 LangChain 可用。"""


def tokenize(text: str) -> list[str]:
    """中英混合分词：中文字符 + 双字 bigram，英文按 word。

    与 BM25 版共用同一个分词器，保证两套检索的输入口径一致。
    """
    lowered = text.lower()
    tokens = _WORD.findall(lowered)
    for run in re.findall(r"[一-鿿]+", lowered):
        tokens.append(run)
        if len(run) > 1:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    rank: int = 0


# --------------------------------------------------------------------------
# BM25 对照组（与 retrieval.py 同源，抽到本文件统一管理）
# --------------------------------------------------------------------------


class HybridRetriever:
    """BM25 + 短语覆盖的混合检索器。"""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        import math
        from collections import Counter

        self._math = math
        self._Counter = Counter
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._build_index()

    def _build_index(self) -> None:
        Counter = self._Counter
        self._doc_tokens = [tokenize(c.text) for c in self.chunks]
        self._tf = [Counter(tokens) for tokens in self._doc_tokens]
        self._len = [len(tokens) for tokens in self._doc_tokens]
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0

        df: Counter = Counter()
        for tokens in self._doc_tokens:
            df.update(set(tokens))
        n = len(self.chunks) or 1
        self._idf = {
            term: self._math.log(1 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        math, Counter = self._math, self._Counter
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_counter = Counter(q_tokens)
        scored: list[float] = []

        for idx in range(len(self.chunks)):
            tf = self._tf[idx]
            length = self._len[idx] or 1
            score = 0.0
            for term, q_freq in q_counter.items():
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * length / (self._avg_len or 1))
                score += idf * (freq * (self.k1 + 1) / denom) * (1 + math.log(q_freq))
            for phrase in _phrases(query):
                if phrase in self.chunks[idx].text:
                    score += 0.8
            scored.append(score)

        order = sorted(range(len(scored)), key=lambda i: scored[i], reverse=True)[:top_k]
        return [
            SearchHit(chunk=self.chunks[i], score=round(scored[i], 4), rank=r + 1)
            for r, i in enumerate(order)
            if scored[i] > 0
        ]


def _phrases(query: str) -> list[str]:
    return [run for run in re.findall(r"[一-鿿]{3,}", query)]


# --------------------------------------------------------------------------
# LangChain 切分
# --------------------------------------------------------------------------


def split_with_langchain(doc, chunk_size: int = 320, overlap: int = 64) -> list[Chunk]:
    """用 LangChain 的 RecursiveCharacterTextSplitter 切分。

    选它而不是自己写，是因为它的分隔符优先级策略（段落 → 句子 → 字符）
    在中英混排文本上比 naive 定长切分稳，而且这是生产里真正在用的组件。

    切完必须把 `chunk_index` 重新编号：`cite_ref` 依赖它，
    沿用 LangChain 返回顺序里的位置会导致锚点漂移。
    """
    from langchain_core.documents import Document as LCDocument
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    pieces = splitter.split_text(doc.raw_text)

    chunks: list[Chunk] = []
    for i, text in enumerate(pieces):
        text = text.strip()
        if not text:
            continue
        cjk = len(re.findall(r"[一-鿿]", text))
        chunks.append(
            Chunk(
                doc_id=doc.doc_id,
                chunk_index=i,
                text=text,
                section=_guess_section(text),
                token_estimate=cjk + max(len(text) - cjk, 0) // 4,
            )
        )
    return chunks


_SECTION_PATTERNS = (
    re.compile(r"^#+\s*(.+)$", re.M),
    re.compile(r"^\s*(摘要|abstract|引言|方法|结论|实验|参考文献)\s*[:：]?", re.I | re.M),
)


def _guess_section(text: str) -> str:
    for pattern in _SECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()[:40]
    return ""


# --------------------------------------------------------------------------
# FAISS 向量检索
# --------------------------------------------------------------------------


class _TfidfCharEmbeddings(_LazyEmbeddingsBase):
    """字符 n-gram TF-IDF 向量，作为 FAISS 索引的嵌入。

    这里不用随机哈希嵌入：`DeterministicFakeEmbedding` 的向量之间
    没有任何语义关系，L2 距离全在 1.7 左右，算出来的余弦相似度恒为负、
    被 clamp 成 0，向量检索就退化成"永远拒答"。那样的演示没有意义。

    改用字符 2~3 gram 的 TF-IDF：不需要下载模型、不需要 API Key、
    完全确定性，而且对中文的术语重合有真实区分度
    （"切分策略"和"切分粒度"共享 n-gram，距离会明显更近）。

    换成真embedding 模型时，只要保证 `embed_query` 返回 L2 归一化向量，
    `VectorRetriever` 剩下的逻辑一行都不用改。
    """

    def __init__(self, size: int = 256) -> None:
        import numpy as np

        self._np = np
        self.size = size
        self._vocab: dict[str, int] = {}
        self._idf = np.zeros(0, dtype="float32")

    @staticmethod
    def _grams(text: str) -> list[str]:
        text = re.sub(r"\s+", "", text.lower())
        grams: list[str] = []
        for n in (2, 3):
            grams.extend(text[i : i + n] for i in range(max(len(text) - n + 1, 0)))
        return grams

    def fit(self, texts: list[str]) -> "_TfidfCharEmbeddings":
        """在语料上建词表与 idf。

        必须在建索引之前调用：`embed_documents` 与 `embed_query` 必须
        用同一套词表和 idf，否则查询向量落在另一个空间里，
        FAISS 返回的距离没有意义。
        """
        np = self._np
        self._vocab = {}
        grams_per_doc = [Counter(self._grams(t)) for t in texts]
        for counter in grams_per_doc:
            for g in counter:
                self._vocab.setdefault(g, len(self._vocab))
        n_docs = len(texts)
        self._idf = np.zeros(len(self._vocab), dtype="float32")
        for g, j in self._vocab.items():
            df = sum(1 for counter in grams_per_doc if g in counter)
            self._idf[j] = np.log((1 + n_docs) / (1 + df)) + 1.0
        return self

    def _project(self, counter: Counter) -> "np.ndarray":
        np = self._np
        vec = np.zeros(len(self._vocab), dtype="float32")
        total = sum(counter.values()) or 1
        for g, freq in counter.items():
            j = self._vocab.get(g)
            if j is not None:
                vec[j] = (freq / total) * self._idf[j]
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed_documents(self, texts: list[str]) -> list:
        if not self._vocab:
            self.fit(texts)
        return [self._project(Counter(self._grams(t))).tolist() for t in texts]

    def embed_query(self, text: str):
        return self._project(Counter(self._grams(text))).tolist()


class VectorRetriever:
    """FAISS 向量检索。

    索引用 `langchain_community.vectorstores.FAISS`，切片带
    `cite_ref` / `doc_id` / `chunk_index` 三个 metadata，
    所以向量检索拿回来的片段和 BM25 那边的 `Chunk` 是同一个东西，
    引用逻辑完全不用改。
    """

    def __init__(self, chunks: list[Chunk], backend: str = "faiss") -> None:
        if backend != "faiss":
            raise ValueError(f"未知向量后端: {backend!r}")
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_core.documents import Document as LCDocument
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "向量检索需要先安装依赖：pip install -r requirements.txt"
            ) from exc

        self.chunks = chunks
        self.embeddings = _TfidfCharEmbeddings()
        self.embeddings.fit([c.text for c in chunks])
        docs = [
            LCDocument(
                page_content=c.text,
                metadata={"doc_id": c.doc_id, "chunk_index": c.chunk_index,
                          "cite_ref": c.cite_ref, "section": c.section},
                )
            for c in chunks
        ]
        # distance_strategy="MAX_INNER_PRODUCT"：FAISS 默认用 L2 距离，
        # 而 TF-IDF 向量极度稀疏（数千维里只有几十维非零），
        # 在欧氏空间里任意两个向量都近似正交，L2 距离恒在 1.6~2.0，
        # 算出来的余弦全是负数、检索信号被抹平。
        # 内积等于归一化后的余弦相似度，才是稀疏词袋该用的度量。
        self.store = FAISS.from_documents(docs, self.embeddings,
                                          distance_strategy="MAX_INNER_PRODUCT")

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        pairs = self.store.similarity_search_with_score(query, k=top_k)
        hits: list[SearchHit] = []
        for rank, (doc, score) in enumerate(pairs, start=1):
            idx = doc.metadata["chunk_index"]
            if idx >= len(self.chunks):
                continue
            # 内积策略下 score 已是归一化余弦（向量都是单位向量），直接用
            hits.append(
                SearchHit(chunk=self.chunks[idx],
                          score=round(max(0.0, min(1.0, float(score))), 4), rank=rank)
            )
        return hits


def make_retriever(kind: str, chunks: list[Chunk]):
    """按名字建检索器：`bm25` 或 `vector`。"""
    if kind == "bm25":
        return HybridRetriever(chunks)
    if kind == "vector":
        return VectorRetriever(chunks)
    raise ValueError(f"未知检索器: {kind!r}，可选 bm25 / vector")


__all__ = [
    "SearchHit",
    "HybridRetriever",
    "VectorRetriever",
    "split_with_langchain",
    "make_retriever",
    "tokenize",
]
