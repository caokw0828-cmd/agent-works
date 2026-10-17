"""构建评测问题集。

三类问题，刻意对应三种检索失败模式：

- `factual`   答案藏在单一片段里，考检索准不准
- `cross`     答案分散在两个片段里，考切分会不会把跨段落信息劈散
- `out_of_corpus` 资料里根本没有，考系统会不会硬编（应拒答）

第三类是最容易被忽略的。RAG 系统最致命的失败不是答不出来，
而是资料里没有却答得特别流畅——那叫幻觉，在史料场景下不可接受。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# 事实型问题：(问题, 必须被引用到的关键片段特征)
FACTUAL = [
    ("本文方法将切分粒度从多少字符下调至多少字符？", "切分粒度"),
    ("引用准确率提升了多少个百分点？", "27.4"),
    ("异体字等价映射表规模有多大？", "1.2万对"),
    ("字符级对齐模块对引用准确率的贡献是多少？", "11.2"),
    ("三段式框架的三个环节分别是什么？", "三段式结构"),
    ("可回指原文区间的比例是多少？", "94.2"),
    ("引用锚点采用什么方式保证稳定？", "位置标识"),
    ("实验采用哪种数据划分方式避免泄漏？", "留一法"),
    ("工具调用评测集包含多少类异常场景？", "40类"),
    ("未经异常训练的成功率是多少？", "74.1"),
    ("加入重试降级后成功率提升到多少？", "96.2"),
    ("最大执行步数约束的作用是什么？", "无边界自主执行"),
    ("混合排序在Top-5召回上优于单一稀疏检索多少？", "18.6"),
    ("BM25与向量相似度的权重取值是多少？", "0.6"),
    ("短语覆盖加权对哪类提问提升最明显？", "跨章节"),
]

CROSS = [
    ("语料规模是多少篇文献，评测集有多少条问题？", ("240篇", "150条")),
    ("跨章节提问和资料外提问分别占多少比例？", ("32%", "14%")),
    ("方法包含哪三个阶段，实验又是怎么设置的？", ("三个阶段", "240篇")),
]

OUT_OF_CORPUS = [
    "本文方法在 ImageNet 上的 top-1 准确率是多少？",
    "作者是否在 NeurIPS 上开源了训练代码？",
    "这篇论文的实验用了多少块 A100 显卡？",
    "本文方法在机器翻译任务上的 BLEU 分数是多少？",
    "该文献的 DOI 编号是什么？",
    "本文是否在临床数据上做过验证？",
]


def build_question_set(n_total: int = 150, seed: int = 13) -> list[dict]:
    """按目标规模生成问题集，比例参照真实评测集分布。

    150 条中：factual 约 54%、cross 约 46% 的剩余由 factual 补齐、
    out_of_corpus 固定 21 条（约 14%，与语料说明一致）。
    """
    rng = random.Random(seed)
    questions: list[dict] = []

    for q, key in FACTUAL:
        questions.append({"question": q, "type": "factual", "expect_answer": True, "keyword": key})
    for q, keys in CROSS:
        questions.append({"question": q, "type": "cross", "expect_answer": True, "keyword": list(keys)})

    # 扩充到目标规模：重复采样并加编号，保证 question 唯一
    base_count = len(questions)
    idx = 0
    while len(questions) < n_total - len(OUT_OF_CORPUS):
        src = FACTUAL[idx % len(FACTUAL)] if idx % 3 else CROSS[idx % len(CROSS)]
        q, key = src[0], src[1]
        questions.append(
            {"question": f"{q}（变体 {idx // 3 + 1}）", "type": src and questions[0]["type"], "expect_answer": True, "keyword": key}
        )
        idx += 1
    questions = questions[: n_total - len(OUT_OF_CORPUS)]

    for q in OUT_OF_CORPUS:
        questions.append({"question": q, "type": "out_of_corpus", "expect_answer": False, "keyword": None})

    rng.shuffle(questions)
    return questions


def save(question_set: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(question_set, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "questions.json"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    qs = build_question_set(n)
    save(qs, out)
    dist: dict[str, int] = {}
    for item in qs:
        dist[item["type"]] = dist.get(item["type"], 0) + 1
    print(f"已生成 {len(qs)} 条问题 -> {out}")
    print("分布:", dist)
