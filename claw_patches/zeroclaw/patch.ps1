# ============================================================
# ZeroClaw Avatar Channel Patch - v0.7.4 (PowerShell)
# ============================================================
# Applies the nyxclaw avatar channel patch to a ZeroClaw v0.7.4
# checkout via `git apply`.
#
# Usage:
#   .\patch.ps1 -ZeroClawDir C:\path\to\zeroclaw-v0.7.4
#
# See patch.sh for full description.
# ============================================================

param(
    [Parameter(Mandatory = $true)]
    [string]$ZeroClawDir
)

$ErrorActionPreference = 'Stop'

$PatchDir     = Split-Path -Parent $MyInvocation.MyCommand.Path
$PatchFile    = Join-Path $PatchDir 'zeroclaw-v0.7.4-nyxclaw.patch'
$ExpectedBase = '78fb0a6'  # zeroclaw v0.7.4 release tag commit

if (-not (Test-Path $PatchFile)) {
    Write-Error "patch file not found: $PatchFile"
    exit 1
}

if (-not (Test-Path (Join-Path $ZeroClawDir 'Cargo.toml'))) {
    Write-Error "$ZeroClawDir\Cargo.toml not found. Is this a ZeroClaw checkout?"
    exit 1
}

if (-not (Test-Path (Join-Path $ZeroClawDir '.git'))) {
    # `.git` is a directory in regular checkouts and a file in `git worktree`s.
    # Test-Path matches either.
    Write-Error "$ZeroClawDir is not a git repository. The patch requires git apply."
    Write-Host  "  Clone with: git clone https://github.com/zeroclaw-labs/zeroclaw -b v0.7.4 $ZeroClawDir"
    exit 1
}

Write-Host '============================================================'
Write-Host 'ZeroClaw Avatar Channel Patch - v0.7.4'
Write-Host '============================================================'
Write-Host "Target:  $ZeroClawDir"
Write-Host "Patch:   $PatchFile"
Write-Host ''

$HeadCommit = (git -C $ZeroClawDir rev-parse --short HEAD).Trim()
Write-Host "HEAD:    $HeadCommit"

git -C $ZeroClawDir merge-base --is-ancestor $ExpectedBase HEAD 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Warning "$ExpectedBase (v0.7.4 release tag) is not an ancestor of HEAD."
    Write-Warning 'The patch was generated against v0.7.4 and may not apply cleanly.'
    $ans = Read-Host 'Continue anyway? [y/N]'
    if ($ans -ne 'y' -and $ans -ne 'Y') {
        exit 1
    }
}

Write-Host ''
Write-Host 'Dry-running patch (git apply --check)...'
git -C $ZeroClawDir apply --check $PatchFile
if ($LASTEXITCODE -ne 0) {
    Write-Error 'Patch does not apply cleanly. Aborting before any changes.'
    exit 1
}
Write-Host '  OK - patch applies cleanly.'

Write-Host ''
Write-Host 'Applying patch...'
git -C $ZeroClawDir apply $PatchFile
if ($LASTEXITCODE -ne 0) {
    Write-Error 'Patch failed during apply. Tree may be in a partial state.'
    exit 1
}
Write-Host '  OK - patch applied.'

Write-Host ''
Write-Host '============================================================'
Write-Host 'Patch applied successfully.'
Write-Host ''
Write-Host 'Next steps:'
Write-Host "  1. cd $ZeroClawDir"
Write-Host '  2. cargo build --workspace'
Write-Host '  3. cargo test -p zeroclaw-gateway --lib nyxclaw'
Write-Host '  4. Update playground/AGENTS.md with the {speech, content}'
Write-Host "     response-format guidance (see $PatchDir\README.md)."
Write-Host '  5. Start the gateway and connect nyxclaw to /ws/avatar.'
Write-Host ''
Write-Host 'To revert:'
Write-Host "  git -C $ZeroClawDir apply -R $PatchFile"
Write-Host '============================================================'
