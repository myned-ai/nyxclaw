<#
.SYNOPSIS
    ZeroClaw Avatar Channel Patch - v0.5.0
.DESCRIPTION
    Applies nyxclaw avatar channel patches to a ZeroClaw v0.5.0
    installation. Creates backups of modified files.

    What it does:
      1. Replaces: traits.rs, openai.rs, agent.rs (patched copies)
      2. Adds: nyxclaw.rs (new avatar channel)
      3. Injects: response_format: None into loop_.rs, anthropic.rs,
         reliable.rs (6 locations)
      4. Injects: pub mod nyxclaw into channels/mod.rs
      5. Injects: /ws/avatar route into gateway/mod.rs

    NOTE: You must manually update AGENTS.md (see README.md)
.EXAMPLE
    .\patch.ps1 -ZeroClawDir C:\path\to\zeroclaw
#>

param(
    [Parameter(Mandatory=$true, HelpMessage="Path to the ZeroClaw v0.5.0 directory")]
    [string]$ZeroClawDir
)

$ErrorActionPreference = "Stop"

$PatchDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ZeroClawDir = Resolve-Path $ZeroClawDir -ErrorAction Stop | Select-Object -ExpandProperty Path
$BackupDir = Join-Path $ZeroClawDir ".nyxclaw-patch-backup"

if (-not (Test-Path (Join-Path $ZeroClawDir "Cargo.toml"))) {
    Write-Error "Cargo.toml not found. Is this a ZeroClaw directory?"
    exit 1
}

Write-Host "============================================================"
Write-Host "ZeroClaw Avatar Channel Patch"
Write-Host "============================================================"
Write-Host "Target:  $ZeroClawDir"
Write-Host "Patches: $PatchDir"
Write-Host ""

if (-not (Test-Path $BackupDir)) {
    New-Item -ItemType Directory -Path $BackupDir | Out-Null
}
Write-Host "Backups: $BackupDir"
Write-Host ""

# -- Helpers -------------------------------------------------

function Backup-File {
    param([string]$RelativePath)
    $FullPath = Join-Path $ZeroClawDir $RelativePath
    if (Test-Path $FullPath) {
        $BackupPath = Join-Path $BackupDir $RelativePath
        $BackupFileDir = Split-Path -Parent $BackupPath
        if (-not (Test-Path $BackupFileDir)) {
            New-Item -ItemType Directory -Path $BackupFileDir | Out-Null
        }
        Copy-Item -Path $FullPath -Destination $BackupPath -Force
    }
}

function Patch-File {
    param([string]$SrcRelativePath, [string]$DestRelativePath)
    $FullDest = Join-Path $ZeroClawDir $DestRelativePath
    $FullSrc = Join-Path $PatchDir $SrcRelativePath

    if (-not (Test-Path $FullSrc)) {
        Write-Host "  SKIP  $SrcRelativePath (not found in patch dir)"
        return
    }

    if (Test-Path $FullDest) {
        Backup-File -RelativePath $DestRelativePath
        Write-Host "  REPLACE $DestRelativePath (backup saved)"
    } else {
        $DestFileDir = Split-Path -Parent $FullDest
        if (-not (Test-Path $DestFileDir)) {
            New-Item -ItemType Directory -Path $DestFileDir | Out-Null
        }
        Write-Host "  NEW     $DestRelativePath"
    }
    
    Copy-Item -Path $FullSrc -Destination $FullDest -Force
}

# -- Step 1: File replacements ------------------------------

Write-Host "Step 1: Replacing patched files..."
Write-Host ""

Patch-File -SrcRelativePath "src/providers/traits.rs" -DestRelativePath "src/providers/traits.rs"
Patch-File -SrcRelativePath "src/providers/openai.rs" -DestRelativePath "src/providers/openai.rs"
Patch-File -SrcRelativePath "src/agent/agent.rs" -DestRelativePath "src/agent/agent.rs"
Patch-File -SrcRelativePath "src/channels/nyxclaw.rs" -DestRelativePath "src/channels/nyxclaw.rs"

Write-Host ""

# -- Step 2: Inject response_format: None -------------------
Write-Host "Step 2: Injecting response_format: None into ChatRequest usages..."
Write-Host ""

# loop_.rs
$File = Join-Path $ZeroClawDir "src/agent/loop_.rs"
if (Test-Path $File) {
    $Content = Get-Content -Path $File -Raw
    if ($Content -notmatch "response_format") {
        Backup-File -RelativePath "src/agent/loop_.rs"
        $Content = $Content -replace '((?s)            ChatRequest \{.+?tools: request_tools,)', "`$1`n                response_format: None,"
        Set-Content -Path $File -Value $Content -NoNewline
        Write-Host "  INJECT src/agent/loop_.rs (1 location)"
    }
}

# anthropic.rs
$File = Join-Path $ZeroClawDir "src/providers/anthropic.rs"
if (Test-Path $File) {
    $Content = Get-Content -Path $File -Raw
    if ($Content -notmatch "response_format") {
        Backup-File -RelativePath "src/providers/anthropic.rs"
        $Content = $Content -replace '((?s)let request = ProviderChatRequest \{.+?Some\(&tool_specs\)\r?\n *\},)', "`$1`n            response_format: None,"
        Set-Content -Path $File -Value $Content -NoNewline
        Write-Host "  INJECT src/providers/anthropic.rs (1 location)"
    }
}

# reliable.rs
$File = Join-Path $ZeroClawDir "src/providers/reliable.rs"
if (Test-Path $File) {
    $Content = Get-Content -Path $File -Raw
    if ($Content -notmatch "response_format") {
        Backup-File -RelativePath "src/providers/reliable.rs"
        $Content = $Content -replace '(tools: [^}]*),(\r?\n *\})', "`$1,`n                response_format: None,`$2"
        Set-Content -Path $File -Value $Content -NoNewline
        Write-Host "  INJECT src/providers/reliable.rs (6 locations)"
    }
    
    # stream_chat delegation
    $Content = Get-Content -Path $File -Raw
    if ($Content -notmatch "fn stream_chat\(") {
        if ($Content -notmatch "StreamEvent") {
            $Content = $Content -replace "ChatResponse, StreamChunk,", "ChatResponse, StreamChunk, StreamEvent,"
        }
        
        $Injection = @"
    fn stream_chat(
        &self,
        request: ChatRequest<'_>,
        model: &str,
        temperature: f64,
    ) -> stream::BoxStream<'static, StreamResult<StreamEvent>> {
        // Delegate to the first provider that supports streaming
        for (_name, provider) in &self.providers {
            if provider.supports_streaming() {
                let current_model = match self.model_chain(model).first() {
                    Some(m) => m.to_string(),
                    None => model.to_string(),
                };
                return provider.stream_chat(request, &current_model, temperature);
            }
        }
        // Fallback: default empty Done
        stream::once(async {
            Ok(StreamEvent::Done(ChatResponse {
                text: None,
                tool_calls: Vec::new(),
                usage: None,
                reasoning_content: None,
            }))
        }).boxed()
    }

"@
        $Content = $Content -replace '    fn stream_chat_with_system\(', "$Injection    fn stream_chat_with_system("
        Set-Content -Path $File -Value $Content -NoNewline
        Write-Host "  INJECT src/providers/reliable.rs (stream_chat delegation)"
    }
}

Write-Host ""

# -- Step 3: Inject StreamEvent re-export into providers/mod.rs -
Write-Host "Step 3: Adding StreamEvent re-export to providers/mod.rs..."
Write-Host ""
$ProvidersMod = Join-Path $ZeroClawDir "src/providers/mod.rs"
if (Test-Path $ProvidersMod) {
    $Content = Get-Content -Path $ProvidersMod -Raw
    if ($Content -notmatch "StreamEvent") {
        Backup-File -RelativePath "src/providers/mod.rs"
        $Content = $Content -replace "ToolCall, ToolResultMessage,", "StreamEvent, ToolCall, ToolResultMessage,"
        Set-Content -Path $ProvidersMod -Value $Content -NoNewline
        Write-Host "  INJECT src/providers/mod.rs (added StreamEvent)"
    } else {
        Write-Host "  SKIP  src/providers/mod.rs (StreamEvent already present)"
    }
}
Write-Host ""

# -- Step 4: Add async_stream dependency (for streaming SSE) -
Write-Host "Step 4: Adding async_stream dependency..."
Write-Host ""
$CargoToml = Join-Path $ZeroClawDir "Cargo.toml"
if (Test-Path $CargoToml) {
    $Content = Get-Content -Path $CargoToml -Raw
    if ($Content -notmatch "async-stream") {
        Backup-File -RelativePath "Cargo.toml"
        $Content = $Content -replace '(?m)(^\[dependencies\])', "$1`nasync-stream = `"0.3`""
        Set-Content -Path $CargoToml -Value $Content -NoNewline
        Write-Host "  INJECT Cargo.toml (added async-stream)"
    } else {
        Write-Host "  SKIP  Cargo.toml (async-stream already present)"
    }
}
Write-Host ""

# -- Step 5: Register channel module + route ----------------
Write-Host "Step 5: Registering nyxclaw channel..."
Write-Host ""

# channels/mod.rs
$ChannelsMod = Join-Path $ZeroClawDir "src/channels/mod.rs"
if (Test-Path $ChannelsMod) {
    $Content = Get-Content -Path $ChannelsMod -Raw
    if ($Content -notmatch "pub mod nyxclaw;") {
        Backup-File -RelativePath "src/channels/mod.rs"
        $Content = $Content -replace '(?m)^pub mod notion;', "pub mod notion;`npub mod nyxclaw;"
        Set-Content -Path $ChannelsMod -Value $Content -NoNewline
        Write-Host "  INJECT src/channels/mod.rs (added pub mod nyxclaw)"
    } else {
        Write-Host "  SKIP  src/channels/mod.rs (already present)"
    }
}

# gateway/mod.rs
$GatewayMod = Join-Path $ZeroClawDir "src/gateway/mod.rs"
if (Test-Path $GatewayMod) {
    $Content = Get-Content -Path $GatewayMod -Raw
    if ($Content -notmatch "nyxclaw") {
        Backup-File -RelativePath "src/gateway/mod.rs"
        
        $Content = $Content -replace '(?m)^pub mod ws;', "pub mod ws;`nuse crate::channels::nyxclaw;"
        $Content = $Content -replace '(?m)^(\s*\.route\("/ws/chat".*)$', "`$1`n        // -- WebSocket avatar channel (nyxclaw) --`n        .route(`"/ws/avatar`", get(nyxclaw::handle_ws_nyxclaw))"
        $Content = $Content -replace '(?m)^(.*GET  /ws/chat.*)$', "`$1`n    println!(`"  GET  /ws/avatar - WebSocket avatar channel (nyxclaw)`");"
        
        Set-Content -Path $GatewayMod -Value $Content -NoNewline
        Write-Host "  INJECT src/gateway/mod.rs (import + route + print)"
    } else {
        Write-Host "  SKIP  src/gateway/mod.rs (already present)"
    }
}

Write-Host ""
Write-Host "  NOTE: Update your playground/AGENTS.md manually (see README.md)"
Write-Host ""

# -- Verify --------------------------------------------------
Write-Host "Verifying..."
$Errors = 0

function Check-File {
    param([string]$RelativePath)
    if (-not (Test-Path (Join-Path $ZeroClawDir $RelativePath))) {
        Write-Host "  MISSING: $RelativePath"
        $script:Errors++
    }
}

function Check-Contains {
    param([string]$RelativePath, [string]$Pattern)
    $TargetFile = Join-Path $ZeroClawDir $RelativePath
    if (Test-Path $TargetFile) {
        $Content = Get-Content -Path $TargetFile -Raw
        if ($Content -notmatch $Pattern) {
            Write-Host "  NOT INJECTED: $RelativePath (missing '$Pattern')"
            $script:Errors++
        }
    }
}

Check-File "src/channels/nyxclaw.rs"
Check-File "src/agent/agent.rs"
Check-Contains "src/providers/traits.rs" "response_format"
Check-Contains "src/providers/openai.rs" "response_format"
Check-Contains "src/agent/loop_.rs" "response_format"
Check-Contains "src/providers/anthropic.rs" "response_format"
Check-Contains "src/providers/reliable.rs" "response_format"
Check-Contains "src/channels/mod.rs" "pub mod nyxclaw"
Check-Contains "src/gateway/mod.rs" "nyxclaw"

if ($Errors -eq 0) {
    Write-Host "  All patches verified."
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Patch applied successfully!"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. cd $ZeroClawDir"
    Write-Host "  2. cargo build"
    Write-Host "  3. cargo test"
    Write-Host "  4. cargo run -- gateway"
    Write-Host "  5. Connect nyxclaw to ws://<host>:<port>/ws/avatar"
    Write-Host "============================================================"
} else {
    Write-Host ""
    Write-Error "$Errors verification(s) failed!"
    exit 1
}
