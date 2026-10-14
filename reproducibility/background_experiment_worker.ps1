param(
    [Parameter(Mandatory = $true)]
    [string]$JobDirectory,

    [Parameter(Mandatory = $true)]
    [string]$ExperimentScript,

    [Parameter(Mandatory = $true)]
    [string]$WorkingDirectory,

    [Parameter(Mandatory = $true)]
    [string]$ArgumentsFile
)

$ErrorActionPreference = "Stop"
$stdoutPath = Join-Path $JobDirectory "stdout.log"
$stderrPath = Join-Path $JobDirectory "stderr.log"
$statusPath = Join-Path $JobDirectory "status.json"
$startedAtInstant = (Get-Date).ToUniversalTime()
$startedAt = $startedAtInstant.ToString("o")

function Write-JobStatus {
    param(
        [string]$State,
        [Nullable[int]]$ExitCode,
        [string]$Message
    )
    $payload = [ordered]@{
        state = $State
        experiment_script = $ExperimentScript
        working_directory = $WorkingDirectory
        started_at_utc = $startedAt
        updated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        exit_code = $ExitCode
        message = $Message
        stdout_log = $stdoutPath
        stderr_log = $stderrPath
    }
    $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

Write-JobStatus -State "running" -ExitCode $null -Message "experiment process started"
$argumentPayload = Get-Content -LiteralPath $ArgumentsFile -Raw | ConvertFrom-Json
[string[]]$experimentArguments = @($argumentPayload.arguments)

try {
    # Quote every argument using Windows native-process escaping. Native file
    # redirection avoids PowerShell retaining experiment output until exit.
    function ConvertTo-NativeArgument([string]$Value) {
        $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
        $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
        return '"' + $escaped + '"'
    }
    $nativeArguments = @('-u', $ExperimentScript) + $experimentArguments
    $quotedArguments = ($nativeArguments | ForEach-Object {
        ConvertTo-NativeArgument $_
    }) -join ' '
    $pythonExecutable = (Get-Command python -ErrorAction Stop).Source
    $experimentProcess = Start-Process `
        -FilePath $pythonExecutable `
        -ArgumentList $quotedArguments `
        -WorkingDirectory $WorkingDirectory `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru

    while (-not $experimentProcess.HasExited) {
        $elapsed = [math]::Round(((Get-Date).ToUniversalTime() - $startedAtInstant).TotalMinutes, 1)
        Write-JobStatus `
            -State "running" `
            -ExitCode $null `
            -Message "experiment process running; heartbeat at ${elapsed} minutes"
        Start-Sleep -Seconds 30
        $experimentProcess.Refresh()
    }

    $experimentProcess.WaitForExit()
    $processExitCode = [int]$experimentProcess.ExitCode
    if ($processExitCode -eq 0) {
        Write-JobStatus -State "completed" -ExitCode $processExitCode -Message "experiment completed successfully"
    }
    else {
        Write-JobStatus -State "failed" -ExitCode $processExitCode -Message "experiment returned a nonzero exit code"
    }
}
catch {
    $_ | Out-String | Add-Content -LiteralPath $stderrPath -Encoding UTF8
    Write-JobStatus -State "failed" -ExitCode 1 -Message $_.Exception.Message
}
