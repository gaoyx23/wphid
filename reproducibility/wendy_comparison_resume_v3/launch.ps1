$ErrorActionPreference = 'Stop'
$taskWorkspace = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $taskWorkspace
$taskOutput = Join-Path $taskWorkspace 'artifacts/paper_comparison_resume_v2/gaussian_fresh_confirmation'
$taskReceipt = Join-Path $taskOutput 'detached-launch.json'
if (Test-Path -LiteralPath $taskReceipt) { throw 'Already launched; do not silently retry' }
$taskPython = (Get-Command python).Source
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList @('-B', 'reproducibility/wendy_comparison_resume_v3/complete.py', 'run') -WorkingDirectory $taskWorkspace -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskOutput 'detached-stdout.txt') -RedirectStandardError (Join-Path $taskOutput 'detached-stderr.txt') -PassThru
@{pid=$taskProcess.Id; python=$taskPython; working_directory=$taskWorkspace; output_root=$taskOutput; started_utc=(Get-Date).ToUniversalTime().ToString('o'); detached=$true} | ConvertTo-Json | Set-Content -LiteralPath $taskReceipt -Encoding utf8
Get-Content -LiteralPath $taskReceipt
