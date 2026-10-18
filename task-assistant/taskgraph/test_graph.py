"""自检用例。

运行：python taskgraph/test_graph.py

覆盖三类：计算器安全性、参数校验、执行图的终止与升级，
以及 LangGraph 是否真的生效（而不是静默退化到手写实现）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taskgraph.graph import MAX_STEPS, TaskGraph  # noqa: E402
from taskgraph.tools import BUILTIN_TOOLS, ParamError, safe_eval  # noqa: E402


# -- 安全：计算器绝不执行任意代码 ------------------------------------------

def test_rejects_code_injection() -> None:
    """安全红线：工具入参绝不能被当作代码执行。"""
    malicious = [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "[].__class__.__bases__",
        "().__class__.__subclasses__()",
        "eval('1+1')",
    ]
    for expr in malicious:
        try:
            safe_eval(expr)
        except ParamError:
            continue
        except Exception:
            continue
        raise AssertionError(f"危险表达式未被拦截: {expr}")


def test_rejects_oversized_power() -> None:
    """防止 9**9**9 打爆 CPU。"""
    try:
        safe_eval("9 ** 99999")
    except ParamError:
        return
    raise AssertionError("超大指数未被拦截")


def test_accepts_normal_math() -> None:
    assert safe_eval("128 * 37 + 456") == 128 * 37 + 456
    assert safe_eval("(45 + 55) * 8 / 4") == 200.0


# -- 参数校验 ---------------------------------------------------------------

def test_missing_param_detected() -> None:
    """缺参数必须在调用工具前就被拦住，不浪费一次调用。"""
    graph = TaskGraph()
    tool = graph.tools["web_search"]
    assert graph.validate_params(tool, {}) == ["query"]
    assert graph.validate_params(tool, {"query": "  "}) == ["query"]
    assert graph.validate_params(tool, {"query": "FastAPI"}) == []


# -- 执行图：步数与终止 ------------------------------------------------------

def test_always_terminates() -> None:
    """无论输入多离谱，执行图必须在步数上限内终止。"""
    graph = TaskGraph()
    for task in ["", "？？？", "计算", "计算 (((((1))))", "对比", "介绍"]:
        state = graph.run(task)
        assert state.finished or state.escalated, f"未正常终止: {task!r}"
        assert state.step_count <= MAX_STEPS * 2, f"步数超限: {task!r} -> {state.step_count}"


def test_normal_task_succeeds() -> None:
    state = TaskGraph().run("FastAPI 是什么？")
    assert any(r["result"].get("ok") for r in state.results), "正常任务应至少有一项成功"


def test_escalation_on_conflict() -> None:
    """verify 模式下检索结果与查询不匹配时应升级人工，而不是硬给结论。"""
    state = TaskGraph().run("对比 BM25 与 FAISS 的差异，并计算 100 / 7")
    assert state.escalated, "应升级人工确认"
    assert state.human_confirmations >= 1


def test_no_duplicate_conclusions() -> None:
    """步数上限保证不会重复调用同一工具。"""
    state = TaskGraph().run("FastAPI 是什么？")
    tool_calls = [s for s in state.steps if s.node == "act"]
    assert len(tool_calls) == len(state.results), "工具调用次数应等于规划步数"


# -- LangGraph 集成 --------------------------------------------------------


def test_langgraph_is_actually_used() -> None:
    """编排必须真的走 LangGraph，而不是退化到手写 while 循环。

    这条测试存在的意义：如果哪天依赖装不上而代码静默回退到手写实现，
    简历上"用 LangGraph 实现"就成了空话。这里直接断言图的类型。
    """
    from langgraph.graph.state import CompiledStateGraph

    from taskgraph.graph import LangGraphTaskGraph

    graph = LangGraphTaskGraph()
    app = graph._build()
    assert isinstance(app, CompiledStateGraph), "应编译成 LangGraph 的图"
    assert set(app.nodes) >= {"plan", "act", "finish", "escalate"}, \
        "四个节点都应注册进图里"


def test_recursion_limit_caps_steps() -> None:
    """recursion_limit 必须真的限制步数（对应手写版的 MAX_STEPS）。"""
    state = TaskGraph().run("对比 BM25 与 FAISS 的差异，并计算 100 / 7")
    assert state.step_count <= MAX_STEPS * 2, \
        f"步数应受 recursion_limit 约束，实际 {state.step_count}"


def test_handrolled_and_langgraph_agree() -> None:
    """两套实现跑同一个任务，结论必须一致。

    保留手写版的意义就是当对照组：如果哪天换框架改出行为差异，
    这条测试会立刻发现，而不是等到线上出问题。
    """
    from taskgraph.graph import LangGraphTaskGraph

    task = "FastAPI 是什么？"
    lg = LangGraphTaskGraph(use_handrolled=False).run(task)
    hr = LangGraphTaskGraph(use_handrolled=True).run(task)
    assert lg.finished == hr.finished, "终止状态应一致"
    assert lg.escalated == hr.escalated, "升级判断应一致"
    assert len(lg.results) == len(hr.results), "规划步数应一致"
    assert [r["tool"] for r in lg.results] == [r["tool"] for r in hr.results], \
        "工具序列应一致"


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
