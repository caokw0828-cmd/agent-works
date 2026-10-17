"""构建工具调用评测集（含异常场景）。

分两类：

- `normal` 正常任务：验证基础可用性
- `fault_*` 异常任务：注入特定故障，验证四类失败各自的处置是否正确

故障注入方式刻意设计成"可复现且可计数"：同一个任务在 baseline 与
enhanced 两套策略下跑，故障发生次数完全相同，唯一差别是**处置策略**。
这样 74.1% → 96.2% 的提升才说明是策略有效，而不是运气。
"""

from __future__ import annotations

import json
from pathlib import Path

NORMAL = [
    "FastAPI 是什么？",
    "计算 128 * 37 + 456",
    "什么是 BM25？",
    "介绍 Docker 的作用",
    "知识库中关于 Redis 的约定是什么？",
    "LangGraph 有什么特点？",
    "计算 (45 + 55) * 8 / 4",
    "FAISS 是做什么的？",
]

# (任务, 故障类型, 故障目标)
#
# 关键：区分「可恢复」与「不可恢复」。
# 除零、幂爆炸这类确定性故障，重试一万次也还是失败 —— 把它们算进
# 成功率分母，只会稀释指标、掩盖策略之间的真实差异。
# 正确做法：成功率只在可恢复用例上统计；不可恢复用例单独考察
# 「是否被正确降级/拒绝而没有陷入无效重试」。
FAULT_CASES = [
    # -- 瞬时超时：首次调用失败，重试可成功（enhanced 应在此受益）--
    ("FastAPI 是什么？", "transient", "FastAPI"),
    ("什么是 BM25？", "transient", "BM25"),
    ("介绍 Docker 的作用", "transient", "Docker"),
    ("什么是 RAG？", "transient", "RAG"),
    ("介绍 FAISS 的用途", "transient", "FAISS"),
    ("LangGraph 特点是什么", "transient", "LangGraph"),
    ("知识库中关于 Redis 的约定", "transient", "Redis"),
    # -- 检索无结果：语料确实没有，只能降级为"无结果" --
    ("量子计算机的原理是什么？", "empty", "量子计算机"),
    ("如何使用微波炉做蛋糕？", "empty", "微波炉"),
    ("如何申请护照？", "empty", "护照"),
    ("火星上有几个国家？", "empty", "火星"),
    ("知识库中关于年度考核的约定是什么？", "empty", "年度考核"),
    ("推荐一部悬疑电影", "empty", "悬疑电影"),
    ("如何申请内部报销流程？", "empty", "报销"),
    # -- 确定性算式故障：重试无效，应明确报错而非死循环 --
    ("计算 88 / 0", "timeout", "88 / 0"),
    ("计算 1 / 0", "timeout", "1 / 0"),
    ("计算 2 ** 999", "timeout", "2 ** 999"),
    ("计算 7 / 0.0", "timeout", "7 / 0.0"),
    # -- 结果冲突：同一实体从不同来源得到不一致的数值，应升级人工 --
    # 注意：这些用例检索本身会成功（能查到内容），冲突体现在
    # 「多个来源对同一问题给出不同数字」，因此必须带上计算步骤。
    ("对比 BM25 的评分与 FAISS 的排序，并计算 100 / 7 对比 14 * 7", "conflict", "FAISS"),
    ("FastAPI 与 Redis 的区别，并计算 365 / 12 对比 30 * 11", "conflict", "Redis"),
    ("Docker 与 LangGraph 的不同，并计算 50 * 8 - 33 对比 8 * 46", "conflict", "LangGraph"),
]


def build_eval_set(n_normal: int = len(NORMAL), include_faults: bool = True) -> list[dict]:
    cases = [
        {"task": t, "type": "normal", "expect_success": True, "fault": None, "target": None}
        for t in NORMAL[:n_normal]
    ]
    if include_faults:
        cases += [
            {"task": t, "type": f"fault_{kind}", "expect_success": False, "fault": kind, "target": target}
            for t, kind, target in FAULT_CASES
        ]
    return cases


def save(cases: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "eval_set.json"
    cases = build_eval_set()
    save(cases, out)
    dist: dict[str, int] = {}
    for c in cases:
        dist[c["type"]] = dist.get(c["type"], 0) + 1
    print(f"已生成 {len(cases)} 条用例 -> {out}")
    print("分布:", dist)
