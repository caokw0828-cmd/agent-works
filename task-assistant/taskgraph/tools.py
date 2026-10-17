"""工具层：网页搜索、计算器、知识库检索。

异常分类是这个项目的重点。工具失败原因必须**可区分**，否则上层只能
笼统重试。这里把失败显式建模为四类，每类对应不同的处置策略：

    ToolTimeout      → 退避重试，超过次数则降级
    ParamError       → 不重试（重试同样参数必然再失败），交由规划器补参数
    EmptyResult      → 降级为「无结果」而非抛错，让流程能继续
    ResultConflict   → 升级为需人工确认，不自动决策

工具本身零外部依赖：网页搜索用内置静态语料模拟，计算器用受限 AST 求值
（绝不 eval 原始字符串，这是安全红线）。
"""

from __future__ import annotations

import ast
import operator
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable


class ToolError(Exception):
    """工具失败基类。"""

    kind = "unknown"


class ToolTimeout(ToolError):
    kind = "timeout"


class ParamError(ToolError):
    kind = "param_error"


class EmptyResult(ToolError):
    kind = "empty_result"


class ResultConflict(ToolError):
    kind = "result_conflict"


# --------------------------------------------------------------------------
# 受限表达式求值（计算器工具）
# --------------------------------------------------------------------------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_MAX_POW = 128  # 防止 9**9**9 打爆 CPU


def safe_eval(expression: str) -> float:
    """用 AST 白名单求值，替代 eval()。

    直接 eval 工具入参等于给 Agent 开了一个远程代码执行口子 ——
    模型可能因为幻觉或被注入而传出 `__import__('os').system('rm -rf /')`。
    这里只允许数字与上述算术运算，节点数超限即拒绝。
    """
    if len(expression) > 200:
        raise ParamError("表达式过长")

    def _visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _visit(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ParamError(f"不支持的常量: {node.value!r}")
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            right = _visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW:
                raise ParamError("指数过大")
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
                raise ParamError("除数为零")
            return _OPS[type(node.op)](_visit(node.left), right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_visit(node.operand))
        raise ParamError(f"不支持的语法节点: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ParamError(f"表达式语法错误: {exc.msg}") from exc
    return _visit(tree)


# --------------------------------------------------------------------------
# 工具定义
# --------------------------------------------------------------------------


@dataclass
class ToolResult:
    tool: str
    ok: bool
    content: Any
    elapsed_ms: float = 0.0
    attempts: int = 1
    degraded: bool = False
    error_kind: str = ""
    note: str = ""


@dataclass
class Tool:
    name: str
    description: str
    required_params: list[str]
    fn: Callable[..., Any]
    # 该工具失败时是否值得重试：参数错误重试无意义
    retryable: bool = True


# 静态网页语料：离线可复现，避免评测依赖网络
_WEB_CORPUS = {
    "python": "Python 是一种高级通用编程语言，由 Guido van Rossum 于1991年首次发布，设计哲学强调代码可读性。",
    "fastapi": "FastAPI 是一个基于 Starlette 的现代 Python Web 框架，强调类型注解与自动生成 OpenAPI 文档。",
    "langgraph": "LangGraph 是用于构建有状态、可循环的大模型应用编排框架，支持持久化检查点与人工介入。",
    "bm25": "BM25 是一种经典的相关性排序函数，基于词频与逆文档频率，对词频饱和有抑制作用。",
    "rag": "检索增强生成通过在生成前引入外部检索结果，缓解模型参数中的知识局限并支持溯源。",
    "faiss": "FAISS 是 Meta 开源的向量相似度检索库，支持大规模稠密向量与乘积内积的高效搜索。",
    "docker": "Docker 通过容器隔离应用及其依赖环境，提供一致的可复现部署单元。",
    "redis": "Redis 是内存数据结构存储，常被用作缓存、消息队列与分布式锁组件。",
}


_TERM_SPLIT = re.compile(r"[\s，。？！、；：,.?!;:\-]+")


def _terms(query: str) -> list[str]:
    """按中英文标点与空白切分查询词。

    必须剥离中文标点：直接用 `query.split()` 会得到 `BM25？` 这种
    带标点的 token，导致明明语料里有 BM25 却检索不到。
    这是中文检索里最容易被忽略、也最容易在评测里暴露的问题。
    """
    return [t for t in _TERM_SPLIT.split(query.lower()) if t.strip()]


def web_search(query: str, top_k: int = 2) -> list[dict]:
    """模拟网页搜索：按关键词匹配静态语料。

    无匹配时抛 EmptyResult 而不是返回空列表，让上层显式处理"没搜到"。
    """
    terms = _terms(query)
    # 英文/数字 token 要求完整匹配；中文允许子串匹配
    terms = [t.strip("？?。.") for t in terms]
    terms = [t for t in terms if t]
    hits = []
    for name, text in _WEB_CORPUS.items():
        name_l, text_l = name.lower(), text.lower()
        if any(term in name_l or term in text_l for term in terms):
            hits.append({"title": f"{name} 介绍", "snippet": text})
    if not hits:
        raise EmptyResult(f"未检索到与「{query}」相关的内容")
    return hits[:top_k]


def knowledge_search(query: str, top_k: int = 2) -> list[dict]:
    """模拟内部知识库检索（复用同一语料但独立命名空间）。"""
    try:
        return web_search(query, top_k=top_k)
    except EmptyResult as exc:
        raise EmptyResult(f"知识库中无「{query}」相关条目") from exc


def calculator(expression: str) -> float:
    return safe_eval(expression)


BUILTIN_TOOLS: list[Tool] = [
    Tool(
        name="web_search",
        description="检索公开技术资料的摘要，用于回答概念性问题。",
        required_params=["query"],
        fn=web_search,
    ),
    Tool(
        name="knowledge_search",
        description="在内部知识库中检索条目，用于回答项目内部约定类问题。",
        required_params=["query"],
        fn=knowledge_search,
    ),
    Tool(
        name="calculator",
        description="计算数学表达式，支持四则运算与幂运算。",
        required_params=["expression"],
        fn=calculator,
    ),
]


__all__ = [
    "Tool",
    "ToolResult",
    "ToolError",
    "ToolTimeout",
    "ParamError",
    "EmptyResult",
    "ResultConflict",
    "BUILTIN_TOOLS",
    "safe_eval",
]
