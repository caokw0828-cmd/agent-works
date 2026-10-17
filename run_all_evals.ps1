<#
一键跑全部评测 + 自检（Windows PowerShell）。
零第三方依赖：不需要 pip install，不需要 API Key。
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

Invoke-Step "1/4  lit-qa-agent 自检"      "lit-qa-agent\litqa\test_agent.py"
Invoke-Step "2/4  lit-qa-agent 评测"      "lit-qa-agent\eval\evaluate.py"
Invoke-Step "3/4  task-assistant 自检"    "task-assistant\taskgraph\test_graph.py"
Invoke-Step "4/4  task-assistant 评测"    "task-assistant\eval\evaluate.py"

Write-Host ""
Write-Host "全部完成。报告位于各项目的 eval_results\ 目录。" -ForegroundColor Green
