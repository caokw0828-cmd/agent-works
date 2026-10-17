"""检索层。

默认用纯 Python 实现的 TF-IDF + BM25 混合打分，保证零依赖可跑；
装上 `rank_bm25` / `faiss` 后可通过 `RetrieverBackend` 切换，
但**评测指标口径不变**——这样对比实验才有意义。

为什么不用 embedding：简历里写的是「RAG 全链路」，面试官真正想看的是
你有没有把「切分 → 检索 → 组装 → 引用」这条链路的每个环节都量化过。
BM25 反而更容易暴露切分策略的缺陷。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from .chunking import Chunk

_CJK = re.compile(r"[一-鿿]")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    """中英混合分词：中文字符 + 双字 bigram，英文按 word。"""
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


class HybridRetriever:
    """BM25 + 短语覆盖的混合检索器。"""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._build_index()

    def _build_index(self) -> None:
        self._doc_tokens = [tokenize(c.text) for c in self.chunks]
        self._tf = [Counter(tokens) for tokens in self._doc_tokens]
        self._len = [len(tokens) for tokens in self._doc_tokens]
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 0.0

        df: Counter = Counter()
        for tokens in self._doc_tokens:
            df.update(set(tokens))
        n = len(self.chunks) or 1
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
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
            # 短语覆盖加成：query 里的连续中文片段在片段中出现则加权
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


__all__ = ["HybridRetriever", "SearchHit", "tokenize"]
