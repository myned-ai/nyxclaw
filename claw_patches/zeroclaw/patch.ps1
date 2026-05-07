# ============================================================
# ZeroClaw Avatar Channel Patch - v0.7.4 (PowerShell)
# ============================================================
# Overlays the patched files from .\files\ onto a ZeroClaw v0.7.4
# checkout.
#
# Usage:
#   .\patch.ps1 -ZeroClawDir C:\path\to\zeroclaw-v0.7.4
#
# See patch.sh for the full description.
# ============================================================

param(
    [Parameter(Mandatory = $true)]
    [string]$ZeroClawDir
)

$ErrorActionPreference = 'Stop'

$PatchDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$FilesDir = Join-Path $PatchDir 'files'

if (-not (Test-Path $FilesDir)) {
    Write-Error "$FilesDir does not exist - patch directory is incomplete."
    exit 1
}

if (-not (Test-Path (Join-Path $ZeroClawDir 'Cargo.toml'))) {
    Write-Error "$ZeroClawDir\Cargo.toml not found. Is this a ZeroClaw checkout?"
    exit 1
}

# Light version sanity-check.
$cargoToml = Get-Content (Join-Path $ZeroClawDir 'Cargo.toml') -Raw
if ($cargoToml -notmatch 'version = "0\.7\.' -and $cargoToml -notmatch '0\.7\.4') {
    Write-Warning "$ZeroClawDir\Cargo.toml does not mention version 0.7.x."
    Write-Warning "The files in this patch were generated against v0.7.4 and may not fit other versions."
    Write-Warning "Continuing anyway..."
}

# Git-clean check (only if target is a git tree).
$IsGitTree = Test-Path (Join-Path $ZeroClawDir '.git')
if ($IsGitTree) {
    $status = git -C $ZeroClawDir status --porcelain 2>$null
    if ($status) {
        Write-Error "$ZeroClawDir has uncommitted changes. Refusing to overlay - you'd lose the ability to revert. Commit or stash first."
        exit 1
    }
}

Write-Host '============================================================'
Write-Host 'ZeroClaw Avatar Channel Patch - v0.7.4'
Write-Host '============================================================'
Write-Host "Target:  $ZeroClawDir"
Write-Host "Source:  $FilesDir"
Write-Host ''

if ($IsGitTree) {
    $head = (git -C $ZeroClawDir rev-parse --short HEAD).Trim()
    $branch = (git -C $ZeroClawDir branch --show-current 2>$null)
    if (-not $branch) { $branch = '(detached)' }
    Write-Host "HEAD:    $head"
    Write-Host "Branch:  $branch"
    Write-Host ''
}

Write-Host 'Files to overlay:'
Get-ChildItem -Path $FilesDir -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($FilesDir.Length + 1)
    Write-Host "  $rel"
}
Write-Host ''

# Robocopy preserves attributes and supports recursive overlay; fall
# back to Copy-Item if Robocopy isn't available (rare on Windows).
if (Get-Command robocopy -ErrorAction SilentlyContinue) {
    # /E = subdirs incl. empty, /NJH /NJS = no header/summary noise,
    # /NFL /NDL = no per-file/dir log, /NP = no progress.
    robocopy $FilesDir $ZeroClawDir /E /NJH /NJS /NFL /NDL /NP | Out-Null
    # Robocopy returns 0-7 for success states; treat as success.
    if ($LASTEXITCODE -gt 7) {
        Write-Error "robocopy failed with exit $LASTEXITCODE"
        exit 1
    }
} else {
    Copy-Item -Path "$FilesDir\*" -Destination $ZeroClawDir -Recurse -Force
}

Write-Host '============================================================'
Write-Host 'Patch applied successfully.'
Write-Host ''
Write-Host 'Next steps:'
Write-Host "  1. cd $ZeroClawDir"
Write-Host '  2. docker compose up -d --build         # recommended'
Write-Host '     OR: cargo build --workspace'
Write-Host "  3. Inside the container:"
Write-Host '       docker exec zeroclaw sed -i ''s/require_pairing = false/require_pairing = true/'' /zeroclaw-data/.zeroclaw/config.toml'
Write-Host '       docker exec zeroclaw zeroclaw config set providers.fallback openai'
Write-Host '       docker restart zeroclaw'
Write-Host '  4. Generate the bearer token (see README.md -> Get the bearer token)'
Write-Host '  5. Update playground/AGENTS.md with the {speech, content}'
Write-Host '     response-format guidance (see README.md)'
Write-Host ''
if ($IsGitTree) {
    Write-Host 'To revert:'
    Write-Host "  git -C $ZeroClawDir checkout -- ."
}
Write-Host '============================================================'
