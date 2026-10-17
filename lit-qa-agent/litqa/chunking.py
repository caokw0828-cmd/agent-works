"""文档加载与切分。

关键点：切分必须携带 `doc_id` 与 `chunk_index`，因为「回答可回指到原文
稳定区间」这个能力完全依赖这两个字段。丢元数据的切分等于不可追溯。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

_PARA_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    """一个可检索、可引用的文本片段。"""

    doc_id: str
    chunk_index: int
    text: str
    page: int = 1
    section: str = ""
    token_estimate: int = 0

    @property
    def cite_ref(self) -> str:
        """引用锚点，形如 `doc_id#p3#c12`。

        这是"生成结论精确回指原文稳定文本区间"的最小实现：
        片段内容变化不会影响这个标识（因为它是位置而非内容哈希）。
        """
        return f"{self.doc_id}#p{self.page}#c{self.chunk_index}"

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "page": self.page,
            "section": self.section,
            "text": self.text,
            "cite_ref": self.cite_ref,
        }


@dataclass
class Document:
    doc_id: str
    title: str
    raw_text: str
    meta: dict = field(default_factory=dict)


def load_documents(data_dir: Path) -> list[Document]:
    """从目录读取 .txt/.md 文献。

    真实 PDF 用 pypdf 解析后转成同样的 Document 结构即可，
    这里保留 txt 以保证仓库零依赖可跑。
    """
    docs: list[Document] = []
    for path in sorted(data_dir.glob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        doc_id = hashlib.sha1(path.name.encode()).hexdigest()[:8]
        title = text.splitlines()[0].lstrip("# ").strip() if text.strip() else path.stem
        docs.append(Document(doc_id=doc_id, title=title, raw_text=text, meta={"path": str(path)}))
    return docs


def split_into_chunks(
    doc: Document,
    chunk_size: int = 320,
    overlap: int = 64,
) -> list[Chunk]:
    """按段落聚合 + 滑动窗口切分。

    保留 `overlap` 的原因：答案常跨句子边界，硬切会把关键信息劈成两半，
    导致"检索到了但答不上"。这是后续做切分策略对比实验的对照组。
    """
    chunks: list[Chunk] = []
    paragraphs = [p.strip() for p in _PARA_SPLIT.split(doc.raw_text) if p.strip()]
    buffer: list[str] = []
    buffer_len = 0
    index = 0

    def flush() -> None:
        nonlocal index, buffer, buffer_len
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        if text:
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    chunk_index=index,
                    text=text,
                    section=_guess_section(text),
                    token_estimate=_estimate(text),
                )
            )
            index += 1
        # 保留尾部 overlap 字符作为下一片的前缀
        tail = text[-overlap:] if overlap else ""
        buffer = [tail] if tail else []
        buffer_len = len(tail)

    for para in paragraphs:
        # 超长段落先硬切，防止单片突破 chunk_size
        while len(para) > chunk_size:
            flush()
            para = para[chunk_size - overlap :]
        if buffer_len + len(para) > chunk_size:
            flush()
        buffer.append(para)
        buffer_len += len(para)

    if buffer:
        text = "\n".join(buffer).strip()
        if text and (not chunks or chunks[-1].text != text):
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    chunk_index=index,
                    text=text,
                    section=_guess_section(text),
                    token_estimate=_estimate(text),
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


def _estimate(text: str) -> int:
    cjk = len(re.findall(r"[一-鿿]", text))
    return cjk + max(len(text) - cjk, 0) // 4


__all__ = ["Chunk", "Document", "load_documents", "split_into_chunks"]
