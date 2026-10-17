"""统一评测报告格式。

两个项目共用一套报告结构，面试时可以直接横向对比，
也避免每个人手搓一个 report.json 导致口径不一致。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MetricResult:
    name: str
    value: float
    unit: str = ""
    higher_is_better: bool = True
    note: str = ""


@dataclass
class EvalReport:
    """一次评测运行的完整结果。"""

    project: str
    backend: str
    dataset_size: int
    metrics: list[MetricResult] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    duration_sec: float = 0.0
    notes: list[str] = field(default_factory=list)

    def add(
        self,
        name: str,
        value: float,
        unit: str = "",
        higher_is_better: bool = True,
        note: str = "",
    ) -> None:
        self.metrics.append(MetricResult(name, value, unit, higher_is_better, note))

    @property
    def headline(self) -> dict[str, float]:
        return {m.name: m.value for m in self.metrics}

    def to_markdown(self) -> str:
        lines = [
            f"# 评测报告 · {self.project}",
            "",
            f"- 后端：`{self.backend}`",
            f"- 数据集规模：{self.dataset_size}",
            f"- 运行时长：{self.duration_sec:.2f}s",
        ]
        if self.config:
            lines += ["", "## 配置", "", "```json", json.dumps(self.config, ensure_ascii=False, indent=2), "```"]
        lines += ["", "## 指标", "", "| 指标 | 数值 | 方向 | 说明 |", "| --- | --- | --- | --- |"]
        for m in self.metrics:
            arrow = "↑" if m.higher_is_better else "↓"
            unit = f" {m.unit}" if m.unit else ""
            lines.append(f"| {m.name} | {m.value}{unit} | {arrow} | {m.note} |")
        if self.notes:
            lines += ["", "## 备注", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "report.json"
        md_path = out_dir / "report.md"
        json_path.write_text(
            json.dumps(
                {
                    "project": self.project,
                    "backend": self.backend,
                    "dataset_size": self.dataset_size,
                    "duration_sec": self.duration_sec,
                    "config": self.config,
                    "metrics": [asdict(m) for m in self.metrics],
                    "notes": self.notes,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        return md_path


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


__all__ = ["MetricResult", "EvalReport", "fmt_pct"]
