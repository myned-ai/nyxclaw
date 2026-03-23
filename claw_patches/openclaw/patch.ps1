<#
.SYNOPSIS
    OpenClaw Avatar SSE Patch
.DESCRIPTION
    Adds /v1/chat/completions/avatar endpoint for nyxclaw integration.

    Usage:
      .\patch.ps1 -OpenClawDir C:\path\to\openclaw-v2026.3.13
#>

param(
    [Parameter(Mandatory=$true, HelpMessage="Path to the OpenClaw directory")]
    [string]$OpenClawDir
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$OpenClawDir = Resolve-Path $OpenClawDir -ErrorAction Stop | Select-Object -ExpandProperty Path
$BackupDir = Join-Path $OpenClawDir ".nyxclaw-patch-backup"

Write-Host "============================================================"
Write-Host "OpenClaw Avatar SSE Patch"
Write-Host "============================================================"
Write-Host "Target:  $OpenClawDir"
Write-Host "Patches: $ScriptDir"
Write-Host ""

# -- Validate ---------------------------------------------------
if (-not (Test-Path (Join-Path $OpenClawDir "src/gateway/openai-http.ts"))) {
    Write-Error "src/gateway/openai-http.ts not found. Is this a valid OpenClaw source directory?"
    exit 1
}

if (-not (Test-Path (Join-Path $OpenClawDir "src/gateway/server-http.ts"))) {
    Write-Error "src/gateway/server-http.ts not found."
    exit 1
}

Write-Host "Backups: $BackupDir"
Write-Host ""

# -- Step 1: Copy avatar-http.ts --------------------------------
Write-Host "Step 1: Installing avatar-http.ts..."
Write-Host ""

$BackupGatewayDir = Join-Path $BackupDir "src/gateway"
if (-not (Test-Path $BackupGatewayDir)) {
    New-Item -ItemType Directory -Path $BackupGatewayDir | Out-Null
}

$ServerHttpPath = Join-Path $OpenClawDir "src/gateway/server-http.ts"
$BackupServerHttpPath = Join-Path $BackupGatewayDir "server-http.ts"

# Backup server-http.ts
if (-not (Test-Path $BackupServerHttpPath)) {
    Copy-Item -Path $ServerHttpPath -Destination $BackupServerHttpPath -Force
    Write-Host "  BACKUP src/gateway/server-http.ts"
}

# Copy new avatar handler
$SrcAvatarHttp = Join-Path $ScriptDir "src/gateway/avatar-http.ts"
$DestAvatarHttp = Join-Path $OpenClawDir "src/gateway/avatar-http.ts"
Copy-Item -Path $SrcAvatarHttp -Destination $DestAvatarHttp -Force
Write-Host "  NEW    src/gateway/avatar-http.ts"

Write-Host ""

# -- Step 2: Inject import + route into server-http.ts ----------
Write-Host "Step 2: Registering avatar route in server-http.ts..."
Write-Host ""

$Content = Get-Content -Path $ServerHttpPath -Raw

# 2a: Add import (after the openai-http import)
if ($Content -notmatch "handleAvatarHttpRequest") {
    $ImportTarget = 'import \{ handleOpenAiHttpRequest \} from "\./openai-http\.js";?'
    $ImportReplacement = "import { handleOpenAiHttpRequest } from `"./openai-http.js`";`nimport { handleAvatarHttpRequest } from `"./avatar-http.js`";"
    $Content = $Content -replace $ImportTarget, $ImportReplacement
    Write-Host "  INJECT import { handleAvatarHttpRequest } from `"./avatar-http.js`""
} else {
    Write-Host "  SKIP  import (already present)"
}

# 2b: Add route stage (before the openai stage)
if ($Content -notmatch '"avatar"') {
    $AvatarStage = @"
      if (openAiChatCompletionsEnabled) {
        requestStages.push({
          name: "avatar",
          run: () =>
            handleAvatarHttpRequest(req, res, {
              auth: resolvedAuth,
              config: openAiChatCompletionsConfig,
              trustedProxies,
              allowRealIpFallback,
              rateLimiter,
            }),
        });
      }
"@
    
    # We replace the first occurrence of `if (openAiChatCompletionsEnabled) {` with our stage + the original `if`
    $Target = "(\s*if\s*\(\s*openAiChatCompletionsEnabled\s*\)\s*\{)"
    # Replace only first occurrence 
    $Content = [regex]::new($Target).Replace($Content, "`n$AvatarStage`n`$1", 1)
    
    Write-Host "  INJECT route stage 'avatar' (before 'openai' stage)"
} else {
    Write-Host "  SKIP  route stage (already present)"
}

Set-Content -Path $ServerHttpPath -Value $Content -NoNewline

Write-Host ""

# -- Verify -----------------------------------------------------
Write-Host "Verifying..."

$Errors = 0

if (-not (Test-Path $DestAvatarHttp)) {
    Write-Host "  FAIL  avatar-http.ts not found"
    $Errors++
} else {
    Write-Host "  OK    avatar-http.ts"
}

$Content = Get-Content -Path $ServerHttpPath -Raw

if ($Content -notmatch "handleAvatarHttpRequest") {
    Write-Host "  FAIL  avatar import not in server-http.ts"
    $Errors++
} else {
    Write-Host "  OK    avatar import in server-http.ts"
}

if ($Content -notmatch '"avatar"') {
    Write-Host "  FAIL  avatar route stage not in server-http.ts"
    $Errors++
} else {
    Write-Host "  OK    avatar route stage in server-http.ts"
}

if ($Errors -gt 0) {
    Write-Host ""
    Write-Error "$Errors verification(s) failed."
    exit 1
}

Write-Host "  All patches verified."
Write-Host ""
Write-Host "============================================================"
Write-Host "Patch applied successfully!"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. cd $OpenClawDir"
Write-Host "  2. npm run build   (or: pnpm build)"
Write-Host "  3. npm start -- gateway --bind lan"
Write-Host "  4. Connect nyxclaw to http://<host>:<port>/v1/chat/completions/avatar"
Write-Host "============================================================"
