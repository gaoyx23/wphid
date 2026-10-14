param(
    [Parameter(Mandatory = $true)]
    [string]$Script,

    [string[]]$ScriptArguments = @(),

    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[A-Za-z0-9_.-]+$")]
    [string]$JobName
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$experimentRoot = (Resolve-Path -LiteralPath (Join-Path $repositoryRoot "experiments")).Path
$candidateScript = if ([System.IO.Path]::IsPathRooted($Script)) {
    $Script
}
else {
    Join-Path $repositoryRoot $Script
}
$resolvedScript = (Resolve-Path -LiteralPath $candidateScript).Path
$experimentPrefix = $experimentRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if (-not $resolvedScript.StartsWith($experimentPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Background runner accepts only scripts inside the experiments directory."
}
if ([System.IO.Path]::GetExtension($resolvedScript) -ne ".py") {
    throw "Background experiment must be a Python script."
}

$jobsRoot = Join-Path $repositoryRoot "artifacts\_background_jobs"
$jobDirectory = Join-Path $jobsRoot $JobName
if (Test-Path -LiteralPath $jobDirectory) {
    throw "Background job directory already exists: $jobDirectory"
}
New-Item -ItemType Directory -Path $jobDirectory -Force | Out-Null

$argumentsFile = Join-Path $jobDirectory "arguments.json"
@{ arguments = @($ScriptArguments) } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $argumentsFile -Encoding UTF8
$worker = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "background_experiment_worker.ps1")).Path
$startupStdout = Join-Path $jobDirectory "worker-startup.stdout.log"
$startupStderr = Join-Path $jobDirectory "worker-startup.stderr.log"

$workerArguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", ('"{0}"' -f $worker),
    "-JobDirectory", ('"{0}"' -f $jobDirectory),
    "-ExperimentScript", ('"{0}"' -f $resolvedScript),
    "-WorkingDirectory", ('"{0}"' -f $repositoryRoot),
    "-ArgumentsFile", ('"{0}"' -f $argumentsFile)
)
$powerShellExecutable = (Get-Process -Id $PID).Path
$process = Start-Process `
    -FilePath $powerShellExecutable `
    -ArgumentList $workerArguments `
    -WindowStyle Hidden `
    -RedirectStandardOutput $startupStdout `
    -RedirectStandardError $startupStderr `
    -PassThru

$statusFile = Join-Path $jobDirectory "status.json"
for ($attempt = 0; $attempt -lt 120; $attempt++) {
    if (Test-Path -LiteralPath $statusFile) {
        break
    }
    $process.Refresh()
    if ($process.HasExited) {
        break
    }
    Start-Sleep -Milliseconds 250
}
if (-not (Test-Path -LiteralPath $statusFile)) {
    $process.Refresh()
    $startupError = if (Test-Path -LiteralPath $startupStderr) {
        (Get-Content -LiteralPath $startupStderr -Raw).Trim()
    }
    else {
        ""
    }
    $detail = if ($startupError) { $startupError } else { "no startup error was captured" }
    $exitDescription = if ($process.HasExited) { $process.ExitCode } else { "still running" }
    throw "Background worker did not reach running state (exit=$exitDescription): $detail"
}

$launcher = [ordered]@{
    job_name = $JobName
    worker_pid = $process.Id
    powershell_executable = $powerShellExecutable
    experiment_script = $resolvedScript
    arguments = @($ScriptArguments)
    working_directory = $repositoryRoot
    launched_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    status_file = $statusFile
    stdout_log = (Join-Path $jobDirectory "stdout.log")
    stderr_log = (Join-Path $jobDirectory "stderr.log")
    worker_startup_stdout_log = $startupStdout
    worker_startup_stderr_log = $startupStderr
}
$launcher | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $jobDirectory "launcher.json") -Encoding UTF8
$launcher | ConvertTo-Json -Depth 4
