#!/usr/bin/env bash
# 一键跑全部自检 + 评测。
#
# 项目二用 LangGraph 编排、项目一用 LangChain 切分，所以需要先装依赖。
# 缺依赖时自动装一次；不需要 API Key。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"

echo "=============================================="
echo " 0/4  检查依赖"
echo "=============================================="
if ! "$PY" -c "import langgraph, langchain" >/dev/null 2>&1; then
  echo "未检测到 LangChain / LangGraph，正在安装 requirements.txt ..."
  "$PY" -m pip install -q -r "$ROOT/requirements.txt"
fi
"$PY" - <<'PY'
import importlib
missing = []
for m in ("langgraph", "langchain", "langchain_core"):
    try:
        importlib.import_module(m)
    except ImportError:
        missing.append(m)
print("依赖就绪" if not missing else f"缺少依赖: {missing}")
PY

echo ""
echo "=============================================="
echo " 1/4  lit-qa-agent 自检"
echo "=============================================="
"$PY" "$ROOT/lit-qa-agent/litqa/test_agent.py"

echo ""
echo "=============================================="
echo " 2/4  lit-qa-agent 评测（切分策略 + 检索方式对照）"
echo "=============================================="
"$PY" "$ROOT/lit-qa-agent/eval/evaluate.py"

echo ""
echo "=============================================="
echo " 3/4  task-assistant 自检"
echo "=============================================="
"$PY" "$ROOT/task-assistant/taskgraph/test_graph.py"

echo ""
echo "=============================================="
echo " 4/4  task-assistant 评测（异常处置对照）"
echo "=============================================="
"$PY" "$ROOT/task-assistant/eval/evaluate.py"

echo ""
echo "全部完成。报告位于各项目的 eval_results/ 目录。"
