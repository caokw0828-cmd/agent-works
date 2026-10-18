<#
一键跑全部自检 + 评测（Windows PowerShell）。
项目二用 LangGraph、项目一用 LangChain，需要先装依赖；不需要 API Key。
#>
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Invoke-Step($title, $script) {
    Write-Host ""
    Write-Host "==============================================" -ForegroundColor Cyan
    Write-Host " $title" -ForegroundColor Cyan
    Write-Host "==============================================" -ForegroundColor Cyan
    & $py (Join-Path $root $script)
}

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " 0/4  检查依赖" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
& $py -c "import langgraph, langchain" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "未检测到 LangChain / LangGraph，正在安装 requirements.txt ..." -ForegroundColor Yellow
    & $py -m pip install -q -r (Join-Path $root "requirements.txt")
}
Write-Host "依赖就绪" -ForegroundColor Green

Invoke-Step "1/4  lit-qa-agent 自检"   "lit-qa-agent\litqa\test_agent.py"
Invoke-Step "2/4  lit-qa-agent 评测"   "lit-qa-agent\eval\evaluate.py"
Invoke-Step "3/4  task-assistant 自检" "task-assistant\taskgraph\test_graph.py"
Invoke-Step "4/4  task-assistant 评测" "task-assistant\eval\evaluate.py"

Write-Host ""
Write-Host "全部完成。报告位于各项目的 eval_results\ 目录。" -ForegroundColor Green
