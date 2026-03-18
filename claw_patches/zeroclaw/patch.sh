#!/bin/bash
# ============================================================
# ZeroClaw Avatar Channel Patch — v0.5.0
# ============================================================
# Applies nyxclaw avatar channel patches to a ZeroClaw v0.5.0
# installation. Creates backups of modified files.
#
# Usage:
#   ./patch.sh /path/to/zeroclaw-v0.5.0
#
# What it does:
#   1. Replaces: traits.rs, openai.rs, agent.rs (patched copies)
#   2. Adds: nyxclaw.rs (new avatar channel)
#   3. Injects: response_format: None into loop_.rs, anthropic.rs,
#      reliable.rs (6 locations)
#   4. Injects: pub mod nyxclaw into channels/mod.rs
#   5. Injects: /ws/avatar route into gateway/mod.rs
#
# NOTE: You must manually update AGENTS.md (see README.md)
# ============================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path-to-zeroclaw-v0.5.0>"
    echo ""
    echo "Example: $0 /home/user/zeroclaw"
    exit 1
fi

ZEROCLAW_DIR="$1"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKUP_DIR="${ZEROCLAW_DIR}/.nyxclaw-patch-backup"

# Validate target
if [ ! -f "${ZEROCLAW_DIR}/Cargo.toml" ]; then
    echo "ERROR: ${ZEROCLAW_DIR}/Cargo.toml not found. Is this a ZeroClaw directory?"
    exit 1
fi

echo "============================================================"
echo "ZeroClaw Avatar Channel Patch"
echo "============================================================"
echo "Target:  ${ZEROCLAW_DIR}"
echo "Patches: ${PATCH_DIR}"
echo ""

# Create backup directory
mkdir -p "${BACKUP_DIR}"
echo "Backups: ${BACKUP_DIR}"
echo ""

# ── Helpers ─────────────────────────────────────────────────

# Copy a patched file (full replacement)
patch_file() {
    local src="$1"
    local dest="$2"
    local full_dest="${ZEROCLAW_DIR}/${dest}"
    local full_src="${PATCH_DIR}/${src}"

    if [ ! -f "${full_src}" ]; then
        echo "  SKIP  ${src} (not found in patch dir)"
        return
    fi

    if [ -f "${full_dest}" ]; then
        local backup_path="${BACKUP_DIR}/${dest}"
        mkdir -p "$(dirname "${backup_path}")"
        cp "${full_dest}" "${backup_path}"
        echo "  REPLACE ${dest} (backup saved)"
    else
        mkdir -p "$(dirname "${full_dest}")"
        echo "  NEW     ${dest}"
    fi

    cp "${full_src}" "${full_dest}"
}

# Backup a file before injecting
backup_file() {
    local file="$1"
    local full_path="${ZEROCLAW_DIR}/${file}"
    if [ -f "${full_path}" ]; then
        local backup_path="${BACKUP_DIR}/${file}"
        mkdir -p "$(dirname "${backup_path}")"
        cp "${full_path}" "${backup_path}"
    fi
}

# ── Step 1: File replacements ──────────────────────────────

echo "Step 1: Replacing patched files..."
echo ""

patch_file "src/providers/traits.rs" "src/providers/traits.rs"
patch_file "src/providers/openai.rs" "src/providers/openai.rs"
patch_file "src/agent/agent.rs" "src/agent/agent.rs"
patch_file "src/channels/nyxclaw.rs" "src/channels/nyxclaw.rs"

echo ""

# ── Step 2: Inject response_format: None ───────────────────
# ChatRequest from traits.rs now requires response_format.
# These files construct ChatRequest but weren't replaced.

echo "Step 2: Injecting response_format: None into ChatRequest usages..."
echo ""

# loop_.rs — 1 location (line ~2458)
FILE="${ZEROCLAW_DIR}/src/agent/loop_.rs"
if [ -f "${FILE}" ] && ! grep -q "response_format" "${FILE}"; then
    backup_file "src/agent/loop_.rs"
    sed -i.bak '/^            ChatRequest {/{
        N;N
        s/\(tools: request_tools,\)/\1\n                response_format: None,/
    }' "${FILE}"
    rm -f "${FILE}.bak"
    echo "  INJECT src/agent/loop_.rs (1 location)"
fi

# anthropic.rs — 1 location (line ~666)
FILE="${ZEROCLAW_DIR}/src/providers/anthropic.rs"
if [ -f "${FILE}" ] && ! grep -q "response_format" "${FILE}"; then
    backup_file "src/providers/anthropic.rs"
    sed -i.bak '/let request = ProviderChatRequest {/{
        N;N;N;N;N;N
        s/\(Some(&tool_specs)\n            },\)/\1\n            response_format: None,/
    }' "${FILE}"
    rm -f "${FILE}.bak"
    echo "  INJECT src/providers/anthropic.rs (1 location)"
fi

# reliable.rs — 6 locations
FILE="${ZEROCLAW_DIR}/src/providers/reliable.rs"
if [ -f "${FILE}" ] && ! grep -q "response_format" "${FILE}"; then
    backup_file "src/providers/reliable.rs"
    # Add response_format: None after every `tools:` line inside ChatRequest blocks
    # Match pattern: "tools: Some(" or "tools: None," followed by closing brace
    sed -i.bak '/ChatRequest {/{
        :loop
        N
        /}/!b loop
        s/\(tools: [^}]*\),\(\n *}\)/\1,\n                response_format: None,\2/g
    }' "${FILE}"
    rm -f "${FILE}.bak"
    echo "  INJECT src/providers/reliable.rs (6 locations)"
fi

echo ""

# ── Step 3: Register channel module + route ────────────────

echo "Step 3: Registering nyxclaw channel..."
echo ""

# channels/mod.rs — add module declaration
CHANNELS_MOD="${ZEROCLAW_DIR}/src/channels/mod.rs"
if [ -f "${CHANNELS_MOD}" ]; then
    if ! grep -q "pub mod nyxclaw;" "${CHANNELS_MOD}"; then
        backup_file "src/channels/mod.rs"
        sed -i.bak '/^pub mod notion;/a\
pub mod nyxclaw;' "${CHANNELS_MOD}"
        rm -f "${CHANNELS_MOD}.bak"
        echo "  INJECT src/channels/mod.rs (added pub mod nyxclaw)"
    else
        echo "  SKIP  src/channels/mod.rs (already present)"
    fi
fi

# gateway/mod.rs — add import + route + print line
GATEWAY_MOD="${ZEROCLAW_DIR}/src/gateway/mod.rs"
if [ -f "${GATEWAY_MOD}" ]; then
    if ! grep -q "nyxclaw" "${GATEWAY_MOD}"; then
        backup_file "src/gateway/mod.rs"
        # Import
        sed -i.bak '/^pub mod ws;/a\
use crate::channels::nyxclaw;' "${GATEWAY_MOD}"
        rm -f "${GATEWAY_MOD}.bak"
        # Route
        sed -i.bak '/\.route("\/ws\/chat"/a\
        // ── WebSocket avatar channel (nyxclaw) ──\
        .route("/ws/avatar", get(nyxclaw::handle_ws_nyxclaw))' "${GATEWAY_MOD}"
        rm -f "${GATEWAY_MOD}.bak"
        # Print
        sed -i.bak '/GET  \/ws\/chat/a\
    println!("  GET  /ws/avatar — WebSocket avatar channel (nyxclaw)");' "${GATEWAY_MOD}"
        rm -f "${GATEWAY_MOD}.bak"
        echo "  INJECT src/gateway/mod.rs (import + route + print)"
    else
        echo "  SKIP  src/gateway/mod.rs (already present)"
    fi
fi

echo ""
echo "  NOTE: Update your playground/AGENTS.md manually (see README.md)"
echo ""

# ── Verify ──────────────────────────────────────────────────

echo "Verifying..."
ERRORS=0

check_file() {
    if [ ! -f "${ZEROCLAW_DIR}/$1" ]; then
        echo "  MISSING: $1"
        ERRORS=$((ERRORS + 1))
    fi
}

check_contains() {
    if [ -f "${ZEROCLAW_DIR}/$1" ] && ! grep -q "$2" "${ZEROCLAW_DIR}/$1"; then
        echo "  NOT INJECTED: $1 (missing '$2')"
        ERRORS=$((ERRORS + 1))
    fi
}

check_file "src/channels/nyxclaw.rs"
check_file "src/agent/agent.rs"
check_contains "src/providers/traits.rs" "response_format"
check_contains "src/providers/openai.rs" "response_format"
check_contains "src/agent/loop_.rs" "response_format"
check_contains "src/providers/anthropic.rs" "response_format"
check_contains "src/providers/reliable.rs" "response_format"
check_contains "src/channels/mod.rs" "pub mod nyxclaw"
check_contains "src/gateway/mod.rs" "nyxclaw"

if [ $ERRORS -eq 0 ]; then
    echo "  All patches verified."
    echo ""
    echo "============================================================"
    echo "Patch applied successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. cd ${ZEROCLAW_DIR}"
    echo "  2. cargo build"
    echo "  3. cargo test"
    echo "  4. cargo run -- gateway"
    echo "  5. Connect nyxclaw to ws://<host>:<port>/ws/avatar"
    echo "============================================================"
else
    echo ""
    echo "ERROR: ${ERRORS} verification(s) failed!"
    exit 1
fi
