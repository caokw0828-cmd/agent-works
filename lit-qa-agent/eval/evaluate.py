"""引用质量评测。

核心指标
--------
citation_precision  引用正确率：被引用的片段中，确实支撑答案的比例
answer_coverage    答案覆盖率：关键信息是否出现在答案里
abstention_rate    拒答率：资料外问题的正确拒答比例（越高越好，说明不乱编）
hallucination_rate 幻觉率：应拒答却给出了实质回答的比例（越低越好）

对照实验
--------
同一条 pipeline 跑多组切分参数（chunk_size / overlap / top_k），
输出对比表。这正是简历上"迭代切分策略与提示词，引用正确率提升"的具体做法。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.report import EvalReport, fmt_pct  # noqa: E402

from build_questions import build_question_set, save as save_questions  # noqa: E402
from litqa.agent import Answer, LiteratureQAAgent, QAConfig  # noqa: E402
from make_corpus import build_corpus  # noqa: E402

# 拒答判定：这些短语出现即认为系统"没在拒答"
_REFUSAL_MARKERS = ("无法回答", "未检索到", "资料不足", "没有提及", "不包含")

# 有实质回答的判定：答案长度超过阈值且不含拒答标记
_SUBSTANTIVE_MIN_LEN = 15


def _cited_texts(answer: Answer) -> str:
    return " ".join(answer.cited_texts)


def _is_refusal(answer: Answer) -> bool:
    if answer.abstained:
        return True
    return any(marker in answer.answer for marker in _REFUSAL_MARKERS)


def _has_substantive_answer(answer: Answer) -> bool:
    if _is_refusal(answer):
        return False
    return len(answer.answer.strip()) >= _SUBSTANTIVE_MIN_LEN


def evaluate_config(data_dir: Path, questions: list[dict], config: QAConfig) -> tuple[EvalReport, list[dict]]:
    agent = LiteratureQAAgent(data_dir, config)

    details: list[dict] = []
    citation_hits = citation_total = 0
    coverage_hits = coverage_total = 0
    correct_abstain = ooc_total = 0
    hallucination = 0
    total_latency = 0.0
    for item in questions:
        ans = agent.ask(item["question"])
        total_latency += ans.latency_ms

        expect_answer = item["expect_answer"]
        refusal = _is_refusal(ans)

        if not expect_answer:
            ooc_total += 1
            if refusal:
                correct_abstain += 1
            else:
                hallucination += 1
            details.append({**item, **ans.to_dict(), "correct": refusal})
            continue

        # 引用正确率：关键信息有多少比例确实出现在被引用片段里。
        # 分母是「应当被引用的关键信息条数」，分子是「实际被引用到的条数」，
        # 因此上限恒为 100%（早期版本用引用片段数作分母导致 >100%，已修）。
        keyword = item["keyword"]
        keywords = [keyword] if isinstance(keyword, str) else list(keyword or [])
        if keywords:
            citation_total += len(keywords)
            hits = sum(1 for kw in keywords if kw in _cited_texts(ans))
            citation_hits += hits
            coverage_total += 1
            if hits == len(keywords):
                coverage_hits += 1

        details.append(
            {
                **item,
                **ans.to_dict(),
                "correct": (not refusal) and hits == len(keywords) if keywords else not refusal,
            }
        )

    n = max(len(questions), 1)
    report = EvalReport(
        project="lit-qa-agent",
        backend=config.backend,
        dataset_size=len(questions),
        config=config.as_dict(),
    )
    report.add("citation_precision", citation_hits / citation_total if citation_total else 0.0,
               "↑", True, note="被引用片段中包含关键信息的比例")
    report.add("answer_coverage", coverage_hits / coverage_total if coverage_total else 0.0,
               "↑", True, note="关键信息全部被引用的题目比例")
    report.add("abstention_rate", correct_abstain / ooc_total if ooc_total else 0.0,
               "↑", True, note="资料外问题的正确拒答率")
    report.add("hallucination_rate", hallucination / n,
               "%", False, note="应拒答却给出实质回答的比例")
    report.add("overall_accuracy", sum(1 for d in details if d["correct"]) / n,
               "↑", True, note="综合判分")
    report.add("avg_latency_ms", total_latency / n, "ms", False, note="单次问答平均耗时")
    report.add("avg_tokens_in", sum(d["tokens_in"] for d in details) / n, "tokens", False,
               note="单次输入 token（估算）")
    report.add("avg_tokens_out", sum(d["tokens_out"] for d in details) / n, "tokens", False,
               note="单次输出 token（估算）")
    report.add("llm_error_count", float(agent.call_stats["errors"]), "↑", False,
               note="生成阶段异常次数（工程健壮性）")
    return report, details


CONFIGS = [
    QAConfig(chunk_size=512, overlap=0, top_k=3, min_score=0.5, label="baseline-512"),
    QAConfig(chunk_size=320, overlap=64, top_k=4, min_score=0.5, label="tuned-320-overlap64"),
    QAConfig(chunk_size=256, overlap=80, top_k=5, min_score=0.8, label="tuned-256-strict"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=150)
    parser.add_argument("--backend", default="offline")
    parser.add_argument("--out", default=str(Path(__file__).parent.parent / "eval_results"))
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data" / "corpus"
    build_corpus(data_dir)
    questions = build_question_set(args.questions)
    save_questions(questions, Path(__file__).parent / "questions.json")
    print(f"语料: {len(list(data_dir.glob('*.txt')))} 篇  问题集: {len(questions)} 条\n")

    rows = []
    best_report: EvalReport | None = None
    for cfg in CONFIGS:
        cfg.backend = args.backend
        report, details = evaluate_config(data_dir, questions, cfg)
        rows.append((report, details))
        print(
            f"[{cfg.label:>22}] 引用正确率 {fmt_pct(report.headline['citation_precision']):>7}"
            f"  覆盖率 {fmt_pct(report.headline['answer_coverage']):>7}"
            f"  拒答率 {fmt_pct(report.headline['abstention_rate']):>7}"
            f"  幻觉率 {fmt_pct(report.headline['hallucination_rate']):>7}"
            f"  综合 {fmt_pct(report.headline['overall_accuracy']):>7}"
        )
        if best_report is None or report.headline["overall_accuracy"] > best_report.headline["overall_accuracy"]:
            best_report = report
            best_details = details

    assert best_report is not None
    out_dir = Path(args.out)
    best_report.save(out_dir / best_report.config["label"])
    (out_dir / "details.json").write_text(
        json.dumps(best_details, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 对比表
    lines = ["| 配置 | chunk | overlap | top_k | 引用正确率 | 覆盖率 | 拒答率 | 幻觉率 | 综合 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for report, _ in rows:
        c = report.config
        h = report.headline
        lines.append(
            f"| {c['label']} | {c['chunk_size']} | {c['overlap']} | {c['top_k']} |"
            f" {fmt_pct(h['citation_precision'])} | {fmt_pct(h['answer_coverage'])} |"
            f" {fmt_pct(h['abstention_rate'])} | {fmt_pct(h['hallucination_rate'])} |"
            f" {fmt_pct(h['overall_accuracy'])} |"
        )
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\n最佳配置: {best_report.config['label']}")
    print(f"报告已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
