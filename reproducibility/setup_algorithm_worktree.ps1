param([switch]$Preview)
$ErrorActionPreference = 'Stop'
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$researchRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$target = [System.IO.Path]::GetFullPath((Join-Path $researchRoot '.worktrees/algorithm-dev'))
if (-not $target.StartsWith($researchRoot + [System.IO.Path]::DirectorySeparatorChar)) { throw 'Unsafe worktree target' }
$baselineBranch = 'codex/research-baseline-20261009'
$devBranch = 'codex/algorithm-dev-20261009'
$manifestPath = Join-Path $researchRoot 'reproducibility/workspace-baseline-manifest.json'
$setupPath = Join-Path $researchRoot 'reproducibility/workspace-baseline-setup.json'
$indexPath = Join-Path $researchRoot '.worktrees/baseline-20261009.index'

function Invoke-ResearchGit {
    param([string[]]$GitArgs)
    $answer = & git -c core.excludesFile= -c core.autocrlf=false -C $researchRoot @GitArgs
    if ($LASTEXITCODE -ne 0) { throw "Git failed: $($GitArgs -join ' ')" }
    return $answer
}

$parent = (Invoke-ResearchGit -GitArgs @('rev-parse','HEAD')).Trim()
$oldBranch = (Invoke-ResearchGit -GitArgs @('symbolic-ref','--short','HEAD')).Trim()
$realIndex = Join-Path $researchRoot '.git/index'
$indexBefore = (Get-FileHash -LiteralPath $realIndex -Algorithm SHA256).Hash
$included = [System.Collections.Generic.List[object]]::new()
$excluded = [System.Collections.Generic.List[object]]::new()
$includedBytes = 0L
$excludedBytes = 0L
$allowed = @('.py','.ps1','.sh','.md','.json','.toml','.yaml','.yml','.txt','.tex','.csv','.svg')
$roots = @('src','experiments','reproducibility','tests','docs','artifacts')
$candidates = @()
foreach ($folder in $roots) { $candidates += Get-ChildItem -LiteralPath (Join-Path $researchRoot $folder) -File -Recurse }
$rootNames = @('.gitignore','.gitattributes','AGENTS.md','README.md','START_HERE.md','pyproject.toml')
$candidates += Get-ChildItem -LiteralPath $researchRoot -File
foreach ($file in ($candidates | Sort-Object FullName -Unique)) {
    $relative = $file.FullName.Substring($researchRoot.Length+1).Replace('\','/')
    if ($relative -in @('reproducibility/workspace-baseline-manifest.json','reproducibility/workspace-baseline-setup.json')) { continue }
    if ($relative -match '(^|/)(__pycache__|\.pytest_cache)(/|$)') { continue }
    $reason = $null
    if ($file.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { $reason = 'reparse-point; not copied' }
    elseif ($relative -notmatch '/' -and $file.Name -notin $rootNames) { $reason = 'root export or non-baseline file' }
    elseif ($relative.StartsWith('artifacts/')) {
        if ($relative.StartsWith('artifacts/_background_jobs/')) { $reason = 'runtime job state' }
        elseif ($file.Extension -ne '.json') { $reason = 'large/generated payload; retained in main directory' }
        elseif ($file.Length -gt 2MB) { $reason = 'JSON larger than 2MiB; retained in main directory' }
    }
    elseif ($relative -match '/' -and $file.Extension -notin $allowed) { $reason = 'binary/cache/export outside baseline scope' }
    $entry = [ordered]@{path=$relative; bytes=$file.Length}
    if ($reason) { $entry.reason = $reason; $excluded.Add($entry); $excludedBytes += $file.Length }
    else { $entry.sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLower(); $included.Add($entry); $includedBytes += $file.Length }
}
$summary = [ordered]@{main_directory=$researchRoot; worktree=$target; parent_commit=$parent; main_branch=$oldBranch;
    baseline_branch=$baselineBranch; development_branch=$devBranch; included_files=$included.Count;
    included_MiB=[math]::Round($includedBytes/1MB,2);
    excluded_files=$excluded.Count; excluded_MiB=[math]::Round($excludedBytes/1MB,2)}
if ($Preview) { $summary | ConvertTo-Json; return }
if ((Test-Path -LiteralPath $target) -or (Test-Path -LiteralPath $indexPath) -or
    (Test-Path -LiteralPath $manifestPath) -or (Test-Path -LiteralPath $setupPath)) { throw 'Existing target/checkpoint; inspect instead of overwriting' }
foreach ($branch in @($baselineBranch,$devBranch)) {
    & git -C $researchRoot show-ref --verify --quiet "refs/heads/$branch"
    if ($LASTEXITCODE -eq 0) { throw "Branch already exists: $branch" }
    if ($LASTEXITCODE -ne 1) { throw 'Cannot inspect branch state' }
}
$manifest = [ordered]@{version='WPHID_WORKSPACE_BASELINE_20261009_V1'; created_at=(Get-Date).ToString('o');
    summary=$summary; main_index_sha256_before=$indexBefore; policy='text/source files and result JSON <=2MiB; no payload links';
    manifest_self_hash_omitted=$true; included=$included; excluded=$excluded;
    limitations='Version snapshot, not independent backup/full replay data. Omitted files remain unchanged in main directory.'}
# Generated provenance, not a source/document rewrite.
[System.IO.File]::WriteAllText($manifestPath, ($manifest | ConvertTo-Json -Depth 8)+"`n", [System.Text.UTF8Encoding]::new($false))
New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
$priorIndexEnv = $env:GIT_INDEX_FILE
try {
    $env:GIT_INDEX_FILE = $indexPath
    Invoke-ResearchGit -GitArgs @('read-tree','--empty') | Out-Null
    $paths = @($included | ForEach-Object { $_.path }) + @('reproducibility/workspace-baseline-manifest.json')
    for ($offset=0; $offset -lt $paths.Count; $offset+=100) {
        $last = [math]::Min($offset+99,$paths.Count-1)
        Invoke-ResearchGit -GitArgs (@('add','--') + $paths[$offset..$last]) | Out-Null
    }
    $tree = (Invoke-ResearchGit -GitArgs @('write-tree')).Trim()
    $commit = (Invoke-ResearchGit -GitArgs @('commit-tree',$tree,'-p',$parent,'-m','chore: byte-preserving research baseline for isolated algorithm development')).Trim()
    Invoke-ResearchGit -GitArgs @('update-ref',"refs/heads/$baselineBranch",$commit,'0000000000000000000000000000000000000000') | Out-Null
}
finally { $env:GIT_INDEX_FILE = $priorIndexEnv }
if ((Get-FileHash -LiteralPath $realIndex -Algorithm SHA256).Hash -ne $indexBefore) { throw 'Original index changed' }
if ((Invoke-ResearchGit -GitArgs @('rev-parse','HEAD')).Trim() -ne $parent) { throw 'Original HEAD changed' }
Invoke-ResearchGit -GitArgs @('worktree','add','-b',$devBranch,$target,$commit) | Out-Null
$verified = 0
foreach ($entry in $included) {
    $sourceHash = (Get-FileHash -LiteralPath (Join-Path $researchRoot $entry.path) -Algorithm SHA256).Hash.ToLower()
    $targetHash = (Get-FileHash -LiteralPath (Join-Path $target $entry.path) -Algorithm SHA256).Hash.ToLower()
    if ($sourceHash -ne $entry.sha256 -or $targetHash -ne $entry.sha256) { throw "Byte mismatch: $($entry.path)" }
    $verified++
}
if ((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash -ne
    (Get-FileHash -LiteralPath (Join-Path $target 'reproducibility/workspace-baseline-manifest.json') -Algorithm SHA256).Hash) { throw 'Manifest copy mismatch' }
if ((Get-FileHash -LiteralPath $realIndex -Algorithm SHA256).Hash -ne $indexBefore) { throw 'Original index changed after worktree creation' }
$result = [ordered]@{complete=$true; baseline_commit=$commit; summary=$summary; verified_source_files=$verified;
    verified_manifest=$true; main_index_unchanged=$true; main_HEAD_unchanged=$true;
    scientific_experiments_run=0; pushed=$false; independent_data_backup_created=$false;
    temporary_index_retained=$indexPath; chats_created_or_archived=$false}
[System.IO.File]::WriteAllText($setupPath, ($result | ConvertTo-Json -Depth 8)+"`n", [System.Text.UTF8Encoding]::new($false))
$result | ConvertTo-Json -Depth 8
