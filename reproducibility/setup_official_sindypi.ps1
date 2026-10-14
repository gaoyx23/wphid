param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$target = Join-Path $repoRoot "tmp\pysindy175"
$wheel = Join-Path $PSScriptRoot "vendor\pysindy-1.7.5-py3-none-any.whl"

if (-not (Test-Path -LiteralPath $wheel)) {
    throw "Missing vendored official PySINDy wheel: $wheel"
}
if (Test-Path -LiteralPath $target) {
    throw "Target already exists: $target. Remove it explicitly before rebuilding."
}

New-Item -ItemType Directory -Path $target | Out-Null
& $Python -m pip install --target $target --no-deps `
    "numpy==1.26.4" `
    "scipy==1.11.4" `
    "scikit-learn==1.3.2" `
    "cvxpy==1.4.2" `
    "osqp==0.6.3" `
    "qdldl==0.1.7.post5" `
    "joblib==1.5.3" `
    "threadpoolctl==3.6.0" `
    $wheel

if ($LASTEXITCODE -ne 0) {
    throw "Official SINDy-PI compatibility environment installation failed."
}

Write-Output "Created isolated official PySINDy 1.7.5 environment at $target"
