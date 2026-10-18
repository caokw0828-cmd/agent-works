"""LLM 后端抽象。

设计目标：同一套 Agent 代码，既能接真实大模型 API，也能在没有任何
Key / 网络的环境下跑通完整链路并产出可复现的评测指标。

实现方式：
- `LLMBackend` 协议定义了唯一需要的方法 `complete(prompt) -> str`
- `OfflineExtractiveBackend`：纯本地、确定性（不采样），从给定上下文里抽取式作答
- `OpenAICompatibleBackend`：走 OpenAI 兼容的 /chat/completions 接口

选择后端的唯一入口是 `get_backend()`，业务代码不应直接 import 具体实现。
"""

from __future__ import annotations

import os
import re
from typing import Protocol, Sequence, runtime_checkable

DEFAULT_OFFLINE_MODEL = "offline-extractive-v1"
DEFAULT_ONLINE_MODEL = os.environ.get("AGENT_LLM_MODEL", "gpt-4o-mini")


@runtime_checkable
class LLMBackend(Protocol):
    """所有后端实现的最小契约。"""

    name: str

    def complete(self, prompt: str) -> str:  # pragma: no cover - 协议声明
        ...


def get_backend(kind: str | None = None) -> LLMBackend:
    """按需返回后端实例。

    kind 优先级：显式入参 > 环境变量 AGENT_LLM_BACKEND > 默认 offline。
    显式传 "online" 但缺少 Key 时直接抛错，不做静默降级 —— 评测场景下
    静默降级会让指标失去意义。
    """
    resolved = (kind or os.environ.get("AGENT_LLM_BACKEND") or "offline").lower()

    if resolved == "offline":
        return OfflineExtractiveBackend()
    if resolved == "online":
        return OpenAICompatibleBackend()
    raise ValueError(f"未知后端类型: {kind!r}，可选 offline / online")


# --------------------------------------------------------------------------
# 离线后端
# --------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")
_CJK = re.compile(r"[一-鿿]")


class OfflineExtractiveBackend:
    """确定性的抽取式后端。

    它不是"假装成大模型的玩具"，而是一个**基线**：任何检索增强系统
    都不应该弱于"把最相关的那几段原文直接拼出来"。评测里同时跑 baseline
    和调优版，差值才说明改进是真的。

    作答规则：从 prompt 的 CONTEXT 段里挑出与问题关键词重合度最高的
    若干句，原样拼接。无 Key、无网络、无随机性，跨机器结果完全一致。
    """

    name = DEFAULT_OFFLINE_MODEL

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    @staticmethod
    def _split_context(prompt: str) -> list[str]:
        """从 prompt 中抽出 CONTEXT 区块，按行拆句。"""
        match = re.search(
            r"<CONTEXT>(.*?)</CONTEXT>", prompt, flags=re.S | re.I
        )
        if not match:
            return []
        return [s for s in _SENT_SPLIT.split(match.group(1)) if s.strip()]

    @staticmethod
    def _keywords(question: str) -> set[str]:
        """中英混合的轻量关键词提取。

        中文按 2-gram 切（无分词器依赖），英文/数字按 word 切。
        """
        question = question.strip()
        terms: set[str] = set()

        for token in re.findall(r"[A-Za-z][A-Za-z0-9_+-]*|\d+(?:\.\d+)?", question):
            terms.add(token.lower())

        for run in _CJK.findall(question):
            pass
        cjk_runs = re.findall(r"[一-鿿]{2,}", question)
        for run in cjk_runs:
            if len(run) == 2:
                terms.add(run)
                continue
            for i in range(len(run) - 1):
                terms.add(run[i : i + 2])
        return terms

    def complete(self, prompt: str) -> str:
        question_match = re.search(r"问题[:：]\s*(.+)", prompt)
        question = question_match.group(1).strip() if question_match else ""

        sentences = self._split_context(prompt)
        if not sentences:
            return "（未检索到可用资料，无法作答）"

        keywords = self._keywords(question)
        scored = []
        for idx, sentence in enumerate(sentences):
            lowered = sentence.lower()
            score = sum(lowered.count(term) for term in keywords)
            # 平滑偏置：同等命中时偏好靠前的片段（通常更接近文档主题句）
            scored.append((score, -idx, sentence.strip()))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        top = [item[2] for item in scored[: self.max_sentences] if item[0] > 0]
        if not top:
            top = [item[2] for item in scored[: self.max_sentences]]
        return "".join(top)


# --------------------------------------------------------------------------
# 真实 API 后端
# --------------------------------------------------------------------------


class OpenAICompatibleBackend:
    """OpenAI 兼容接口后端。

    刻意只用标准库的 urllib：评测脚本不因网络客户端库的版本问题而失败，
    安装失败面最小。代理环境下需自行设置 HTTPS_PROXY / NO_PROXY。
    """

    name = DEFAULT_ONLINE_MODEL

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or DEFAULT_ONLINE_MODEL
        self.base_url = (
            base_url or os.environ.get("AGENT_LLM_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("AGENT_LLM_API_KEY")
        self.timeout = timeout
        self.temperature = temperature
        if not self.api_key:
            raise RuntimeError(
                "未提供 API Key：请设置环境变量 AGENT_LLM_API_KEY，"
                "或改用 AGENT_LLM_BACKEND=offline"
            )

    def complete(self, prompt: str) -> str:
        import json
        import urllib.request

        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：CJK 每字约 1 token，其余约 4 字符 1 token。

    只用于成本量级对比，不用于计费。真实计数请用 tiktoken。
    """
    cjk = len(_CJK.findall(text))
    other = max(len(text) - cjk, 0)
    return cjk + other // 4


__all__ = [
    "LLMBackend",
    "OfflineExtractiveBackend",
    "OpenAICompatibleBackend",
    "get_backend",
    "estimate_tokens",
    "Sequence",
]
