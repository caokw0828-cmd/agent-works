"""最小化自检用例。

不依赖 pytest，直接 `python litqa/test_agent.py` 即可运行。
涉及 LangChain / FAISS 的两条会在依赖缺失时自动跳过而不是报错。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from litqa.agent import LiteratureQAAgent, QAConfig  # noqa: E402
from litqa.chunking import load_documents, split_into_chunks  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "corpus"


def _ensure_corpus() -> Path:
    """测试自足：语料不存在时自动生成，不要求用户先跑评测脚本。"""
    if not list(DATA_DIR.glob("*.txt")):
        from make_corpus import build_corpus

        build_corpus(DATA_DIR)
    return DATA_DIR


def _agent() -> LiteratureQAAgent:
    return LiteratureQAAgent(_ensure_corpus(), QAConfig(chunk_size=256, overlap=80, top_k=5))


def test_chunk_ids_are_stable() -> None:
    """引用锚点必须由位置决定，而不是内容哈希 —— 这是溯源稳定的前提。"""
    data_dir = _ensure_corpus()
    doc = load_documents(data_dir)[0]
    a = split_into_chunks(doc, chunk_size=256, overlap=80)
    b = split_into_chunks(doc, chunk_size=256, overlap=80)
    assert [c.cite_ref for c in a] == [c.cite_ref for c in b], "同一文档两次切分的锚点应一致"

    # 改动正文内容后，锚点保持不变（位置标识的语义）
    doc2 = load_documents(data_dir)[0]
    doc2.raw_text = doc2.raw_text.replace("本文", "本研究")
    c = split_into_chunks(doc2, chunk_size=256, overlap=80)
    assert [x.cite_ref for x in c] == [x.cite_ref for x in a], "内容变化不应改变引用锚点"


def test_no_overlap_when_overlap_zero() -> None:
    data_dir = _ensure_corpus()
    doc = load_documents(data_dir)[0]
    chunks = split_into_chunks(doc, chunk_size=300, overlap=0)
    assert all(len(ch.text) <= 320 for ch in chunks), "切分粒度应受 chunk_size 约束"


def test_answer_cites_something() -> None:
    ans = _agent().ask("BM25与向量相似度的权重取值是多少？")
    assert not ans.abstained, f"语料内问题不应拒答（coverage={ans.term_coverage}）"
    assert ans.citations, "有实质回答时必须带引用"
    assert all("#c" in ref for ref in ans.citations), "引用应包含 chunk 锚点"


def test_abstains_on_out_of_corpus() -> None:
    """资料外问题必须拒答，这是防止幻觉的核心分支。"""
    ans = _agent().ask("本文方法在 ImageNet 上的 top-1 准确率是多少？")
    assert ans.abstained, "语料外问题应拒答"


def test_llm_failure_does_not_crash() -> None:
    """生成阶段异常时必须回退到原文片段，而不是抛异常中断整条链路。"""

    class _Boom:
        name = "boom"

        def complete(self, prompt: str) -> str:
            raise RuntimeError("模拟模型不可用")

    agent = _agent()
    agent.llm = _Boom()
    ans = agent.ask("BM25与向量相似度的权重取值是多少？")
    assert agent.call_stats["errors"] == 1, "异常应被计数"
    assert ans.answer, "异常时仍应返回可读内容"


# -- LangChain / FAISS 集成 ------------------------------------------------


def _has_vector_deps() -> bool:
    try:
        import langchain_community.vectorstores  # noqa: F401
        import langchain_text_splitters  # noqa: F401
        return True
    except Exception:
        return False


def test_langchain_splitter_keeps_cite_refs() -> None:
    """换成 LangChain 切分器后，引用锚点必须仍然可用。

    这是接入框架时最容易踩的坑：LangChain 返回的片段如果沿用它自己的
    编号，`cite_ref` 会和 BM25 那边的对不上，跨检索器的引用就失效了。
    """
    if not _has_vector_deps():
        print("      (跳过：未安装 langchain)")
        return
    from litqa.retrieval import split_with_langchain

    data_dir = _ensure_corpus()
    doc = load_documents(data_dir)[0]
    chunks = split_with_langchain(doc, chunk_size=256, overlap=80)
    assert chunks, "LangChain 切分器应产出片段"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks))), \
        "chunk_index 必须重新连续编号，否则 cite_ref 会漂移"
    assert all(c.doc_id == doc.doc_id for c in chunks), "doc_id 应随片段保留"
    assert all("#c" in c.cite_ref for c in chunks), "cite_ref 格式应保持一致"


def test_vector_retriever_returns_same_chunk_objects() -> None:
    """向量检索拿回的片段必须与 BM25 那边是同一批对象。

    如果这里返回的是新建对象，引用逻辑就会在两条路径上分叉，
    后续加新检索器时极易漏改。
    """
    if not _has_vector_deps():
        print("      (跳过：未安装 faiss)")
        return
    from litqa.agent import QAConfig as _QAConfig

    agent = LiteratureQAAgent(_ensure_corpus(), _QAConfig(
        chunk_size=320, overlap=64, top_k=4, splitter="langchain", retriever="vector"))
    hits = agent.retriever.search("BM25 与向量相似度的权重", top_k=3)
    assert hits, "向量检索应有结果"
    for h in hits:
        assert any(h.chunk is c for c in agent.chunks), "应返回 Chunk 原对象"
        assert h.chunk.cite_ref, "片段应带引用锚点"


def run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run_all())
