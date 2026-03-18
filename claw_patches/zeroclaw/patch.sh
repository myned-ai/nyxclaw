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
#   - Copies patched provider files (traits.rs, openai.rs)
#   - Copies patched agent file (agent.rs)
#   - Adds new avatar channel (nyxclaw.rs)
#   - Patches gateway/mod.rs to register /ws/avatar route
#   - Patches channels/mod.rs to register the module
#   - NOTE: You must manually update AGENTS.md (see README.md)
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

# ── Helper ──────────────────────────────────────────────────
patch_file() {
    local src="$1"   # relative path from patch dir
    local dest="$2"  # relative path in zeroclaw dir

    local full_dest="${ZEROCLAW_DIR}/${dest}"
    local full_src="${PATCH_DIR}/${src}"

    if [ ! -f "${full_src}" ]; then
        echo "  SKIP  ${src} (not found in patch dir)"
        return
    fi

    # Backup existing file if it exists
    if [ -f "${full_dest}" ]; then
        local backup_path="${BACKUP_DIR}/${dest}"
        mkdir -p "$(dirname "${backup_path}")"
        cp "${full_dest}" "${backup_path}"
        echo "  PATCH ${dest} (backup saved)"
    else
        mkdir -p "$(dirname "${full_dest}")"
        echo "  NEW   ${dest}"
    fi

    cp "${full_src}" "${full_dest}"
}

# ── Apply patches ───────────────────────────────────────────

echo "Applying patches..."
echo ""

# Provider patches (modified files)
patch_file "src/providers/traits.rs" "src/providers/traits.rs"
patch_file "src/providers/openai.rs" "src/providers/openai.rs"

# Agent patch (modified file)
patch_file "src/agent/agent.rs" "src/agent/agent.rs"

# Avatar channel (new file)
patch_file "src/channels/nyxclaw.rs" "src/channels/nyxclaw.rs"

# ── Inject lines into existing files (instead of replacing) ──

# channels/mod.rs — add module declaration
CHANNELS_MOD="${ZEROCLAW_DIR}/src/channels/mod.rs"
if [ -f "${CHANNELS_MOD}" ]; then
    if ! grep -q "nyxclaw" "${CHANNELS_MOD}"; then
        cp "${CHANNELS_MOD}" "${BACKUP_DIR}/src/channels/mod.rs"
        # Insert after the 'pub mod notion;' line
        sed -i.bak '/^pub mod notion;/a\
pub mod nyxclaw;' "${CHANNELS_MOD}"
        rm -f "${CHANNELS_MOD}.bak"
        echo "  INJECT src/channels/mod.rs (added pub mod nyxclaw)"
    else
        echo "  SKIP  src/channels/mod.rs (nyxclaw already present)"
    fi
fi

# gateway/mod.rs — add import + route + print line
GATEWAY_MOD="${ZEROCLAW_DIR}/src/gateway/mod.rs"
if [ -f "${GATEWAY_MOD}" ]; then
    if ! grep -q "nyxclaw" "${GATEWAY_MOD}"; then
        cp "${GATEWAY_MOD}" "${BACKUP_DIR}/src/gateway/mod.rs"
        # Add import after 'pub mod ws;'
        sed -i.bak '/^pub mod ws;/a\
use crate::channels::nyxclaw;' "${GATEWAY_MOD}"
        rm -f "${GATEWAY_MOD}.bak"
        # Add route after '/ws/chat' route
        sed -i.bak '/\.route("\/ws\/chat"/a\
        // ── WebSocket avatar channel (nyxclaw) ──\
        .route("/ws/avatar", get(nyxclaw::handle_ws_nyxclaw))' "${GATEWAY_MOD}"
        rm -f "${GATEWAY_MOD}.bak"
        # Add print line after '/ws/chat' print
        sed -i.bak '/GET  \/ws\/chat/a\
    println!("  GET  /ws/avatar — WebSocket avatar channel (nyxclaw)");' "${GATEWAY_MOD}"
        rm -f "${GATEWAY_MOD}.bak"
        echo "  INJECT src/gateway/mod.rs (added /ws/avatar route + import)"
    else
        echo "  SKIP  src/gateway/mod.rs (nyxclaw already present)"
    fi
fi

echo ""
echo "  NOTE: Update your playground/AGENTS.md manually (see README.md)"

# ── Verify ──────────────────────────────────────────────────

echo "Verifying patch..."
ERRORS=0

check_file() {
    if [ ! -f "${ZEROCLAW_DIR}/$1" ]; then
        echo "  MISSING: $1"
        ERRORS=$((ERRORS + 1))
    fi
}

check_file "src/providers/traits.rs"
check_file "src/providers/openai.rs"
check_file "src/agent/agent.rs"
check_file "src/channels/nyxclaw.rs"

if [ $ERRORS -eq 0 ]; then
    echo "  All files in place."
    echo ""
    echo "============================================================"
    echo "Patch applied successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. cd ${ZEROCLAW_DIR}"
    echo "  2. cargo build"
    echo "  3. Test: cargo test"
    echo "  4. Run: cargo run -- gateway"
    echo "  5. Connect nyxclaw to ws://<host>:<port>/ws/avatar"
    echo "============================================================"
else
    echo ""
    echo "ERROR: ${ERRORS} file(s) missing after patch!"
    exit 1
fi
