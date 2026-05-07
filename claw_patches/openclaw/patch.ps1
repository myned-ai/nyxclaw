# ============================================================
# OpenClaw Avatar SSE Patch - v2026.5.6 (PowerShell)
# ============================================================
# Overlays the patched files from .\files\ onto an OpenClaw v2026.5.6
# checkout.
#
# Usage:
#   .\patch.ps1 -OpenClawDir C:\path\to\openclaw-v2026.5.6
#
# See patch.sh for the full description.
# ============================================================

param(
    [Parameter(Mandatory = $true)]
    [string]$OpenClawDir
)

$ErrorActionPreference = 'Stop'

$PatchDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$FilesDir = Join-Path $PatchDir 'files'

if (-not (Test-Path $FilesDir)) {
    Write-Error "$FilesDir does not exist - patch directory is incomplete."
    exit 1
}

if (-not (Test-Path (Join-Path $OpenClawDir 'package.json'))) {
    Write-Error "$OpenClawDir\package.json not found. Is this an OpenClaw checkout?"
    exit 1
}

if (-not (Test-Path (Join-Path $OpenClawDir 'src/gateway/server-http.ts'))) {
    Write-Error "$OpenClawDir\src\gateway\server-http.ts not found."
    exit 1
}

# Light version sanity-check.
$pkgJson = Get-Content (Join-Path $OpenClawDir 'package.json') -Raw
if ($pkgJson -notmatch '"version":\s*"2026\.5\.') {
    Write-Warning "$OpenClawDir\package.json does not mention version 2026.5.x."
    Write-Warning "The files in this patch were generated against v2026.5.6 and may not fit other versions."
    Write-Warning "Continuing anyway..."
}

# Git-clean check (only if target is a git tree).
$IsGitTree = Test-Path (Join-Path $OpenClawDir '.git')
if ($IsGitTree) {
    $status = git -C $OpenClawDir status --porcelain 2>$null
    if ($status) {
        Write-Error "$OpenClawDir has uncommitted changes. Refusing to overlay - you'd lose the ability to revert. Commit or stash first."
        exit 1
    }
}

Write-Host '============================================================'
Write-Host 'OpenClaw Avatar SSE Patch - v2026.5.6'
Write-Host '============================================================'
Write-Host "Target:  $OpenClawDir"
Write-Host "Source:  $FilesDir"
Write-Host ''

if ($IsGitTree) {
    $head = (git -C $OpenClawDir rev-parse --short HEAD).Trim()
    $branch = (git -C $OpenClawDir branch --show-current 2>$null)
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
    robocopy $FilesDir $OpenClawDir /E /NJH /NJS /NFL /NDL /NP | Out-Null
    # Robocopy returns 0-7 for success states; treat as success.
    if ($LASTEXITCODE -gt 7) {
        Write-Error "robocopy failed with exit $LASTEXITCODE"
        exit 1
    }
} else {
    Copy-Item -Path "$FilesDir\*" -Destination $OpenClawDir -Recurse -Force
}

Write-Host '============================================================'
Write-Host 'Patch applied successfully.'
Write-Host ''
Write-Host 'Next steps:'
Write-Host "  1. cd $OpenClawDir"
Write-Host '  2. npm install'
Write-Host '  3. npm run build'
Write-Host '  4. npm start -- gateway --bind lan --port 18789'
Write-Host '  5. Set OPENCLAW_GATEWAY_TOKEN in your env (or docker-compose.yml)'
Write-Host '  6. Add {speech, content} response-format guidance to your'
Write-Host '     workspace AGENTS.md (see README.md)'
Write-Host ''
if ($IsGitTree) {
    Write-Host 'To revert:'
    Write-Host "  git -C $OpenClawDir checkout -- ."
}
Write-Host '============================================================'
