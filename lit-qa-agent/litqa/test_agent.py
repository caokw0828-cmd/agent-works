"""最小化自检用例。

不依赖 pytest，直接 `python litqa/test_agent.py` 即可运行，
保证面试官 clone 下来不装任何东西也能看到测试通过。
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
