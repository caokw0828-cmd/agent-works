#!/usr/bin/env bash
# 一键跑全部评测 + 自检。
#
# 刻意做到零第三方依赖：clone 下来直接执行，不需要 pip install，
# 也不需要 API Key。面试现场演示时最怕的就是"我这里装不上"。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"

echo "=============================================="
echo " 1/4  lit-qa-agent 自检"
echo "=============================================="
"$PY" "$ROOT/lit-qa-agent/litqa/test_agent.py"

echo ""
echo "=============================================="
echo " 2/4  lit-qa-agent 评测（切分策略对照）"
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
