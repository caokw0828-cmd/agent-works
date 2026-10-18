"""显式状态与执行图（手写版，作为 LangGraph 实现的对照组）。

`graph.py` 是项目的主实现，用 LangGraph 的 StateGraph 编排。
这个文件保留一套不用框架的等价实现，两者跑同一份评测集，
由 `test_handrolled_and_langgraph_agree` 断言行为一致。

留它的两个理由：
1. 回答"框架到底省了什么" —— 对比 node/edge 与 while 循环的代码量与差异
2. 换框架时如果改出了行为差异，这条对照能立刻发现，而不是等出问题

核心安全约束：**最大步数硬上限**。没有它，一个反复失败的工具调用会
让 Agent 无限循环，在生产上就是一张烧钱的账单。LangGraph 版用框架原生的
`recursion_limit` 实现同一个约束。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .tools import (
    BUILTIN_TOOLS,
    EmptyResult,
    ParamError,
    ResultConflict,
    Tool,
    ToolError,
    ToolResult,
    ToolTimeout,
)

MAX_STEPS = 8

# 步骤决策
Decision = Literal["plan", "act", "reflect", "finish", "escalate"]


@dataclass
class StepRecord:
    step: int
    node: str
    decision: str
    tool: str = ""
    ok: bool = True
    attempts: int = 1
    degraded: bool = False
    error_kind: str = ""
    elapsed_ms: float = 0.0
    note: str = ""


@dataclass
class TaskState:
    """显式状态。每一步都完整落盘，便于事后复盘与回归对比。"""

    task: str
    steps: list[StepRecord] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    step_count: int = 0
    finished: bool = False
    escalated: bool = False
    final_answer: str = ""
    retries: int = 0
    degradations: int = 0
    human_confirmations: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "finished": self.finished,
            "escalated": self.escalated,
            "final_answer": self.final_answer,
            "step_count": self.step_count,
            "retries": self.retries,
            "degradations": self.degradations,
            "human_confirmations": self.human_confirmations,
            "elapsed_ms": round(self.elapsed_ms(), 2),
            "results": self.results,
            "steps": [vars(s) for s in self.steps],
        }


# --------------------------------------------------------------------------
# 任务规划：把自然语言任务拆成工具调用序列
# --------------------------------------------------------------------------

def _entities(query: str) -> list[str]:
    """抽取查询中的实体名词（英文技术词 / 中文实词短语）。

    用于校验检索结果是否真的与查询相关。
    """
    ents = re.findall(r"[A-Za-z][A-Za-z0-9_+]{2,}", query)
    ents += re.findall(r"[一-鿿]{3,}", query)
    return [e for e in ents if e not in ("是什么", "为什么", "怎么", "如何", "什么", "区别", "差异", "对比")]


# 意图规则按优先级从高到低排列，第一个匹配的胜出。：「计算类」必须排在「概念类」之后。反过来会把
# 「量子计算机的原理是什么」误判为计算任务 —— 因为 "原理" 中的
# 字被算术字符类命中，而此时问题里根本抽不出算式。
# 教训：关键词路由里，**更具体（可提取实参）的意图优先级要高于更宽泛的**。
_INTENT_RULES: list[tuple[str, str, str, str]] = [
    # 明确要求计算，且能从问题里抽出算式，才算计算意图
    (r"(?:^|[计算求值算])\s*([\d][\d\s\.\+\-\*/\^\(\)]*[+\-\*/\^\)]\s*[\d])", "calculator", "expression", "计算类任务且可提取算式"),
    (r"(fastapi|docker|redis|faiss|langgraph)", "web_search", "query", "命中已知技术名词"),
    (r"(知识库|内部|约定|规范)", "knowledge_search", "query", "内部知识查询"),
    (r"(什么是|如何|怎么|介绍|区别|原理|是什么)", "web_search", "query", "通用概念查询"),
    # 兜底：只要含算式就算计算（覆盖 "计算" 之外的说法）
    (r"[\d][\d\s\.\+\-\*/\^\(\)]*[+\-\*/\^\)]\s*[\d]", "calculator", "expression", "含算式表达式"),
]


@dataclass
class PlannedCall:
    tool: str
    params: dict[str, str]
    rationale: str


def plan(task: str) -> tuple[list[PlannedCall], str]:
    """规则式规划器：从任务里识别工具序列。

    刻意用规则而非模型：规划结果必须可复现、可审计，评测才有意义。
    真实系统把这一步换成 LLM 即可，后面的执行与异常处置逻辑完全复用。
    """
    calls: list[PlannedCall] = []
    lowered = task.lower()

    # 显式「并 / 然后 / 再」连接复合任务
    segments = [s for s in re.split(r"[，,；;]|\s+然后\s+|\s+再\s+", task) if s.strip()]

    for segment in segments or [task]:
        for pattern, tool, param, rationale in _INTENT_RULES:
            if re.search(pattern, segment, re.I):
                if tool == "calculator":
                    expr = re.search(r"([\d\s\.\+\-\*/\^\(\)]+)", segment)
                    params = {"expression": expr.group(1).strip()} if expr else {}
                else:
                    # 剥掉"计算/介绍/什么是"这类指令词，保留实义内容作为检索词
                    query = re.sub(r"^(计算|算|求值|介绍|什么是|如何|怎么|知识库中关于)\s*", "", segment.strip())
                    params = {"query": (query or segment).strip()}
                calls.append(PlannedCall(tool, params, rationale))
                break

    if not calls:
        calls.append(PlannedCall("web_search", {"query": task.strip()}, "兜底：按通用检索处理"))

    # 结果冲突检测意图：多来源交叉验证
    needs_verify = bool(re.search(r"(对比|交叉|验证|是否一致|对比一下)", task))
    return calls, "verify" if needs_verify else "single"


# --------------------------------------------------------------------------
# 执行图
# --------------------------------------------------------------------------


class TaskGraph:
    """节点 + 条件路由的显式执行图。"""

    def __init__(self, tools: list[Tool] | None = None, max_retries: int = 2,
                 enable_fallback: bool = True) -> None:
        self.tools = {t.name: t for t in (tools or BUILTIN_TOOLS)}
        self.max_retries = max_retries
        self.enable_fallback = enable_fallback
        self.stats = {"calls": 0, "ok": 0, "failed": 0, "escalated": 0}

    # -- 参数校验 ---------------------------------------------------------
    @staticmethod
    def validate_params(tool: Tool, params: dict[str, Any]) -> list[str]:
        missing = [p for p in tool.required_params if p not in params or not str(params[p]).strip()]
        return missing

    # -- 单次工具调用（含重试与降级）--------------------------------------
    def invoke(self, tool: Tool, params: dict[str, Any], allow_retry: bool = True) -> ToolResult:
        started = time.perf_counter()
        missing = self.validate_params(tool, params)
        if missing:
            return ToolResult(
                tool=tool.name, ok=False, content=None,
                error_kind="param_error", attempts=1,
                note=f"缺少参数: {missing}",
            )

        attempts = 0
        max_attempts = (self.max_retries + 1) if allow_retry else 1
        last_error: Exception | None = None
        while attempts < max_attempts:
            attempts += 1
            self.stats["calls"] += 1
            try:
                content = tool.fn(**params)
                return ToolResult(
                    tool=tool.name, ok=True, content=content, attempts=attempts,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                )
            except ParamError as exc:
                # 参数类错误重试无意义（同样的参数必然同样失败），直接交上层补参数
                self.stats["failed"] += 1
                return ToolResult(tool=tool.name, ok=False, content=None, attempts=attempts,
                                  error_kind="param_error", note=str(exc),
                                  elapsed_ms=(time.perf_counter() - started) * 1000)
            except EmptyResult as exc:
                self.stats["failed"] += 1
                # 降级为「无结果」而非失败：流程得以继续，不中断整条链路
                return ToolResult(tool=tool.name, ok=False, content=[], attempts=attempts,
                                  degraded=True, error_kind="empty_result", note=str(exc),
                                  elapsed_ms=(time.perf_counter() - started) * 1000)
            except ResultConflict as exc:
                self.stats["failed"] += 1
                self.stats["escalated"] += 1
                return ToolResult(tool=tool.name, ok=False, content=None, attempts=attempts,
                                  error_kind="result_conflict", note=str(exc),
                                  elapsed_ms=(time.perf_counter() - started) * 1000)
            except ToolTimeout as exc:
                # 瞬时故障（超时/网络抖动）才值得退避重试。
                # 注意：若故障是「确定性」的（同样的参数必然再次失败），
                # 重试纯属浪费——这种情形应由上层的 fallback_to 处理。
                last_error = exc
                time.sleep(0.01 * attempts)
            except Exception as exc:  # noqa: BLE001 兜底：未知异常按瞬时故障处理
                last_error = exc

        self.stats["failed"] += 1
        return ToolResult(tool=tool.name, ok=False, content=None, attempts=attempts,
                          error_kind="timeout", degraded=True,
                          note=f"重试 {attempts} 次仍失败: {last_error}",
                          elapsed_ms=(time.perf_counter() - started) * 1000)

    # -- 降级到备选工具 ---------------------------------------------------
    FALLBACK_MAP = {"web_search": "knowledge_search", "knowledge_search": "web_search"}

    def invoke_with_fallback(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """执行工具，失败时切换到备选工具再试一次。

        这是 enhanced 策略比 baseline 强的关键：baseline 只知道"失败"，
        enhanced 知道"失败的原因"，因此可以换一条路而不是原地重试。
        前提是备选工具在**语义上确实能覆盖**原工具的职责 —— 这里
        web_search 与 knowledge_search 检索同一语料的不同命名空间，
        属于可解释的降级；无覆盖关系时不做降级，直接升级人工。
        """
        primary = self.tools.get(tool_name)
        if primary is None:
            return ToolResult(tool=tool_name, ok=False, content=None, error_kind="unknown_tool")

        result = self.invoke(primary, params)

        if result.ok:
            return result

        fallback_name = self.FALLBACK_MAP.get(tool_name)
        # 结果冲突不降级：冲突意味着"有多个答案"，必须人工裁决
        if not fallback_name or not self.enable_fallback or result.error_kind == "result_conflict":
            return result
        fallback_tool = self.tools.get(fallback_name)
        if fallback_tool is None:
            return result

        self.stats["fallbacks"] = self.stats.get("fallbacks", 0) + 1
        retry_params = dict(params)
        if fallback_name == "calculator":
            retry_params = params
        alt = self.invoke(fallback_tool, retry_params, allow_retry=False)
        alt.tool = f"{fallback_name}(降级自{tool_name})"
        alt.note = f"{result.note}；已降级到 {fallback_name}"
        return alt

    # -- 路由 -------------------------------------------------------------
    def _decide(self, state: TaskState) -> Decision:
        if state.escalated:
            return "escalate"
        if not state.results:
            return "plan"
        pending = [r for r in state.results if not r.get("_visited")]
        if pending:
            return "act"
        # 全部项已处理：走一次汇总即结束，不再空转
        return "finish"

    # -- 主循环 -----------------------------------------------------------
    def run(self, task: str, fault_injector: Callable[[str, str], None] | None = None) -> TaskState:
        state = TaskState(task=task)
        calls, mode = plan(task)
        plan_meta = {c.tool: c for c in calls}
        self.stats = {"calls": 0, "ok": 0, "failed": 0, "escalated": 0}

        while state.step_count < MAX_STEPS * 2:
            decision = self._decide(state)
            state.step_count += 1

            if decision == "escalate":
                state.finished = True
                state.escalated = True
                state.human_confirmations += 1
                state.final_answer = "存在来源冲突，需人工确认后再输出结论。"
                state.steps.append(StepRecord(state.step_count, "escalate", "escalate", ok=False,
                                              note="result_conflict"))
                break

            if decision == "finish":
                state.finished = True
                state.final_answer = self._synthesize(state, mode)
                state.steps.append(StepRecord(state.step_count, "finish", "finish",
                                              note="任务完成" if not state.escalated else "升级人工"))
                break

            if decision == "act":
                entry = next(r for r in state.results if not r.get("_visited"))
                entry["_visited"] = True
                tool_name = entry["tool"]
                tool = self.tools.get(tool_name)
                if tool is None:
                    entry["result"] = {"ok": False, "error_kind": "unknown_tool"}
                    continue

                if fault_injector:
                    fault_injector(tool_name, entry["params"].get("query") or
                                   entry["params"].get("expression") or "")

                result = self.invoke_with_fallback(tool_name, entry["params"])
                entry["result"] = {
                    "ok": result.ok,
                    "content": result.content,
                    "degraded": result.degraded,
                    "error_kind": result.error_kind,
                    "note": result.note,
                }
                if result.ok:
                    self.stats["ok"] += 1
                if result.attempts > 1:
                    state.retries += result.attempts - 1
                if result.degraded:
                    state.degradations += 1
                # 结果冲突升级人工：这类问题自动裁决的风险高于收益
                if result.error_kind == "result_conflict":
                    state.escalated = True
                state.steps.append(
                    StepRecord(state.step_count, "act", "act", tool=result.tool or tool_name,
                               ok=result.ok,
                               attempts=result.attempts, degraded=result.degraded,
                               error_kind=result.error_kind,
                               elapsed_ms=round(result.elapsed_ms, 2),
                               note=result.note)
                )
                continue

            # plan：首次把规划结果写入 state，后续由 _visited 推进
            if decision == "plan":
                state.results = [
                    {"tool": c.tool, "params": c.params, "rationale": c.rationale,
                     "result": {"ok": False}, "_visited": False}
                    for c in calls
                ]
                state.steps.append(StepRecord(state.step_count, "plan", "plan",
                                              note=f"拆解为 {len(calls)} 步（模式={mode}）"))
                continue

            # reflect：汇总校验，不消耗实际工具调用
            state.steps.append(StepRecord(state.step_count, "reflect", "reflect",
                                          note=f"汇总 {len(state.results)} 项结果"))

        for r in state.results:
            r.pop("_visited", None)
            r.pop("rationale", None)
        return state

    def _synthesize(self, state: TaskState, mode: str) -> str:
        """结果汇总与校验。

        `mode == "verify"`（任务要求对比/交叉验证）时执行一致性校验：
        多来源结论不一致则**不强行给单一答案**，而是显式升级人工。
        这是"结果校验"节点的实际作用，也是避免"自动决策掩盖冲突"的关键。
        """
        ok_items = [r for r in state.results if r["result"].get("ok")]

        if mode == "verify" and len(ok_items) >= 2:
            conflict = self._detect_conflict(ok_items)
            if conflict:
                state.escalated = True
                state.human_confirmations += 1
                return f"多来源结论存在冲突（{conflict}），已升级人工确认，不自动输出结论。"

        degraded = [r for r in state.results if r["result"].get("degraded")]
        failed = [r for r in state.results
                  if not r["result"].get("ok") and not r["result"].get("degraded")]

        if not ok_items:
            parts = []
            if degraded:
                parts.append("所需信息未能获取到")
            if failed:
                parts.append(f"{len(failed)} 项工具调用失败")
            return "；".join(parts) + "，无法给出结论。"

        lines = [f"已完成 {len(ok_items)}/{len(state.results)} 项查询。"]
        for item in ok_items:
            content = item["result"]["content"]
            if isinstance(content, list) and content and isinstance(content[0], dict):
                first = content[0]
                lines.append(f"- {item['tool']}：{first.get('snippet') or first.get('content')}")
            elif isinstance(content, (int, float)):
                lines.append(f"- {item['tool']}：{content}")
        if degraded:
            lines.append(f"提示：{len(degraded)} 项查询无结果，已按降级处理。")
        if mode == "verify":
            lines.append("（多来源交叉验证：结论一致）")
        return "\n".join(lines)

    @staticmethod
    def _detect_conflict(ok_items: list[dict]) -> str:
        """结论冲突检测。

        两类冲突：
        1. 数值不一致 —— 多来源返回不同数字
        2. 来源存疑 —— 检索结果与查询实体不匹配（检索器返回了不相干的条目）

        第 2 类在实践中比第 1 类更常见：一个看起来"成功"的检索可能
        返回完全不相关的文档，如果不做一致性校验，Agent 会拿着错误来源
        输出看似笃定的结论。这类错误比直接失败危险得多。
        """
        numbers: list[tuple[str, str]] = []
        for item in ok_items:
            content = item["result"].get("content")
            for num in re.findall(r"\d+(?:\.\d+)?", str(content)):
                numbers.append((item["tool"], num))
        distinct = {n for _, n in numbers}
        if numbers and len(distinct) > 1:
            tools = "/".join(t for t, _ in numbers)
            return f"{tools} 返回数值不一致（{sorted(distinct)}）"

        # 来源存疑：检索结果的标题未出现在查询中，且查询里有实体名词
        for item in ok_items:
            content = item["result"].get("content")
            if not isinstance(content, list):
                continue
            query = str(item["params"].get("query") or "")
            entities = [t for t in _entities(query) if len(t) > 2]
            titles = " ".join(str(c.get("title", "")) for c in content)
            if entities and not any(e.lower() in titles.lower() for e in entities):
                return f"{item['tool']} 返回结果与查询实体不匹配（查询={query}）"
        return ""


__all__ = ["TaskGraph", "TaskState", "StepRecord", "MAX_STEPS", "plan"]
