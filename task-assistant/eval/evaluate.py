"""工具调用评测。

对照实验
--------
同一份用例集、同一套故障注入，跑两套策略：

  baseline  只做一次调用，失败即返回错误（多数 Agent 的默认写法）
  enhanced  启用参数校验 + 分类重试 + 降级 + 人工确认

唯一变量是处置策略，因此成功率差值可直接归因。

指标
----
success_rate        工具调用成功率（简历主指标）
handled_rate        异常被正确分类处置的比例（不含未处理崩溃）
escalation_rate     需人工确认的比例（结果冲突场景应全部升级）
avg_steps           单任务平均步数（步数上限约束是否生效）
avg_latency_ms      单任务平均耗时
retry_count         平均重试次数
crash_count         未捕获异常导致流程中断的次数（必须为 0）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.report import EvalReport, fmt_pct  # noqa: E402

from build_eval_set import build_eval_set, save as save_cases  # noqa: E402
from taskgraph.graph import MAX_STEPS, TaskGraph  # noqa: E402
from taskgraph.tools import EmptyResult, ParamError, Tool, ToolTimeout, safe_eval  # noqa: E402


def make_faulty_tools(base: list[Tool], strategy: str) -> tuple[list[Tool], dict]:
    """构造带故障注入的工具集。

    关键设计：故障是**确定性**的，且按「每条用例」独立计数，
    因此 baseline 与 enhanced 两次运行遇到完全相同的故障序列。
    成功率差值只能归因于处置策略，而非随机波动 —— 这是对照实验成立的前提。

    故障分三类，分别对应三种处置策略：

    - transient  每条用例的首次调用超时 → 退避重试可解决（enhanced 受益）
    - persistent 永远超时/无结果       → 重试无效，靠 fallback 或降级
    - 正常        无故障               → 两策略都应成功（作为对照组）

    注入状态挂在返回的 trigger 字典上，且每次 run_strategy 重建工具集，
    因此 baseline 与 enhanced 每次都从相同起点触发相同故障序列。
    """
    trigger = {"transient_used": set(), "calls": 0, "persistent": 0}
    tools: list[Tool] = []

    TRANSIENT_KWS = ("FastAPI", "BM25", "Docker", "RAG", "FAISS", "LangGraph", "Redis")
    PERSISTENT_KWS = ("量子计算机", "微波炉", "护照", "火星", "年度考核", "悬疑电影", "报销")

    for tool in base:
        original = tool.fn

        def make_fn(fn=original, name=tool.name):
            def wrapper(**kwargs):
                query = str(kwargs.get("query") or kwargs.get("expression") or "")
                trigger["calls"] += 1

                # 1) 确定性算式故障：非法算式，重试必然再失败
                if name == "calculator" and _is_broken_math(query):
                    trigger["persistent"] += 1
                    raise ToolTimeout(f"工具 {name} 超时（确定性故障，重试无效）")

                # 2) 瞬时故障：同一条 query 第一次超时，重试可成功
                if any(k in query for k in TRANSIENT_KWS) and query not in trigger["transient_used"]:
                    trigger["transient_used"].add(query)
                    raise ToolTimeout(f"工具 {name} 瞬时超时")

                # 3) 检索无结果：语料确实没有，fallback 也救不回来
                if any(k in query for k in PERSISTENT_KWS):
                    trigger["persistent"] += 1
                    raise EmptyResult(f"未检索到与「{query}」相关的内容")

                return fn(**kwargs)

            return wrapper

        tools.append(
            Tool(
                name=tool.name,
                description=tool.description,
                required_params=tool.required_params,
                fn=make_fn(),
                retryable=tool.retryable,
            )
        )

    return tools, trigger


def _is_broken_math(query: str) -> bool:
    """判断算式是否属于「确定性失败」类型。

    除零、幂爆炸、超出数值范围这类问题，再重试多少次都会以同样方式失败，
    属于不可恢复故障；合法算式则不应被注入。
    """
    if not any(k in query for k in ("/", "**", "1e308")):
        return False
    try:
        safe_eval(query)
        return False
    except ParamError:
        return True
    except Exception:
        # OverflowError / ZeroDivisionError 等同样视为确定性失败
        return True


def run_strategy(strategy: str, cases: list[dict]) -> tuple[EvalReport, list[dict]]:
    from taskgraph.tools import BUILTIN_TOOLS

    tools, trigger = make_faulty_tools(BUILTIN_TOOLS, strategy)
    max_retries = 2 if strategy == "enhanced" else 0
    # baseline 只调用一次、失败即返回（多数 Agent 的默认写法）；
    # enhanced 启用分类重试 + 降级到备选工具。
    graph = TaskGraph(tools=tools, max_retries=max_retries,
                      enable_fallback=(strategy == "enhanced"))

    details: list[dict] = []
    total_steps = total_latency = total_retries = crashes = 0
    handled = 0
    escalations = expected_escalations = 0
    # 只有「本应成功」的用例参与成功率统计。
    # fault_empty（语料里真的没有内容）与 fault_timeout（确定性算式错误）
    # 本就不可救，把它们算进分母只会稀释指标、掩盖策略差异。
    # 它们改由 handled_rate（是否被正确降级/拒绝）来考察。
    UNRECOVERABLE = {"fault_empty", "fault_timeout"}
    recoverable = [c for c in cases if c["type"] not in UNRECOVERABLE]
    successes = 0

    for case in cases:
        started = time.perf_counter()
        crash = None
        try:
            state = graph.run(case["task"])
        except Exception as exc:  # noqa: BLE001
            crashes += 1
            crash = f"{type(exc).__name__}: {exc}"
            state = None
        elapsed = (time.perf_counter() - started) * 1000

        if state is None:
            details.append({**case, "success": False, "crash": crash,
                            "handled": False, "escalated": False})
            continue

        ok_count = sum(1 for r in state.results if r["result"].get("ok"))
        success = ok_count > 0 and not state.escalated

        # 「被正确处置」= 没崩，且要么成功、要么走了降级/升级路径
        is_handled = success or state.escalated or any(
            r["result"].get("degraded") for r in state.results
        )
        if is_handled:
            handled += 1

        if case in recoverable and success:
            successes += 1

        if case["type"] == "fault_conflict":
            expected_escalations += 1
            if state.escalated:
                escalations += 1

        total_steps += state.step_count
        total_latency += elapsed
        total_retries += state.retries

        details.append({
            **case,
            "success": success,
            "counted": case in recoverable,
            "handled": is_handled,
            "escalated": state.escalated,
            "steps": state.step_count,
            "retries": state.retries,
            "degradations": state.degradations,
            "elapsed_ms": round(elapsed, 2),
            "final_answer": state.final_answer,
            "crash": crash,
        })

    n = len(cases)
    report = EvalReport(
        project="task-assistant",
        backend=f"rule-planner/{strategy}",
        dataset_size=n,
        config={
            "strategy": strategy,
            "max_retries": max_retries,
            "enable_fallback": strategy == "enhanced",
            "max_steps": MAX_STEPS,
            "recoverable_cases": len(recoverable),
        },
    )
    report.add("tool_success_rate", successes / len(recoverable) if recoverable else 0.0,
               "↑", True, note="可恢复用例（正常+瞬时故障）的工具调用成功率")
    report.add("handled_rate", handled / n, "↑", True,
               note="全部用例中异常被正确分类处置（重试/降级/升级）的比例")
    report.add("escalation_rate", escalations / expected_escalations if expected_escalations else 0.0,
               "↑", True, note="结果冲突场景中正确升级人工确认的比例")
    report.add("avg_steps", total_steps / n, "steps", False, note="单任务平均步数")
    report.add("avg_latency_ms", total_latency / n, "ms", False, note="单任务平均耗时")
    report.add("avg_retries", total_retries / n, "times", False, note="平均重试次数")
    report.add("crash_count", float(crashes), "↓", False, note="未捕获异常导致流程中断（必须为 0）")
    report.add("max_steps", float(MAX_STEPS), "↓", False, note="步数硬上限")
    return report, details


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).parent.parent / "eval_results"))
    args = parser.parse_args()

    cases = build_eval_set()
    save_cases(cases, Path(__file__).parent / "eval_set.json")
    print(f"用例集: {len(cases)} 条（正常 {sum(1 for c in cases if c['type']=='normal')} / "
          f"异常 {sum(1 for c in cases if c['type']!='normal')}）\n")

    reports = []
    for strategy in ("baseline", "enhanced"):
        report, details = run_strategy(strategy, cases)
        reports.append((report, details))
        h = report.headline
        print(f"[{strategy:>8}] 成功率 {fmt_pct(h['tool_success_rate']):>7}"
              f"  异常处置率 {fmt_pct(h['handled_rate']):>7}"
              f"  升级率 {fmt_pct(h['escalation_rate']):>7}"
              f"  平均步数 {h['avg_steps']:.2f}"
              f"  耗时 {h['avg_latency_ms']:.1f}ms"
              f"  崩溃 {int(h['crash_count'])}")

    out_dir = Path(args.out)
    for report, _ in reports:
        report.save(out_dir / report.config["strategy"])

    base, enh = reports[0][0].headline, reports[1][0].headline
    lines = [
        "| 指标 | baseline | enhanced | 差值 |",
        "| --- | --- | --- | --- |",
    ]
    for key in ("tool_success_rate", "handled_rate", "avg_retries", "avg_latency_ms", "crash_count"):
        b, e = base[key], enh[key]
        arrow = "↑" if key in ("tool_success_rate", "handled_rate") else "↓"
        lines.append(f"| {key} | {b:.4g} | {e:.4g} | {e - b:+.4g} |")
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines))
    print(f"\n成功率变化: {fmt_pct(base['tool_success_rate'])} → {fmt_pct(enh['tool_success_rate'])}")
    print(f"报告已写入 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
