#!/bin/bash
# ============================================================
# ZeroClaw Avatar Channel Patch — v0.7.4
# ============================================================
# Overlays the patched files from ./files/ onto a ZeroClaw v0.7.4
# checkout. Adds the nyxclaw avatar channel (/ws/avatar) plus
# response_format threading, native OpenAI SSE streaming, the
# size-limit hardening, and the Dockerfile/compose fixes upstream
# v0.7.4 is missing.
#
# Usage:
#   ./patch.sh /path/to/zeroclaw-v0.7.4
#
# What it does:
#   1. Sanity-check the target is ZeroClaw v0.7.4 source
#   2. If the target is a git checkout, refuse if there are uncommitted
#      changes (so a future revert via `git checkout -- .` is clean)
#   3. Copy every file from ./files/ into the target, preserving paths
#   4. Print next-step build/test commands
#
# Reverting:
#   - If the target is a git checkout: `git -C <target> checkout -- .`
#     resets every modified file to its upstream v0.7.4 state.
#   - Otherwise: re-extract the upstream source.
#
# NOTE: After patching you must update playground/AGENTS.md to instruct
# the LLM about the {speech, content} response format. See README.md.
# ============================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path-to-zeroclaw-v0.7.4>"
    echo ""
    echo "Example: $0 /home/user/zeroclaw-v0.7.4"
    exit 1
fi

ZEROCLAW_DIR="$1"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
FILES_DIR="${PATCH_DIR}/files"

if [ ! -d "${FILES_DIR}" ]; then
    echo "ERROR: ${FILES_DIR} does not exist — patch directory is incomplete."
    exit 1
fi

if [ ! -f "${ZEROCLAW_DIR}/Cargo.toml" ]; then
    echo "ERROR: ${ZEROCLAW_DIR}/Cargo.toml not found. Is this a ZeroClaw checkout?"
    exit 1
fi

# Light version sanity-check — bails out if the workspace Cargo.toml
# looks unrelated. Doesn't try to be too clever; the real safety is the
# git-clean check below.
if ! grep -q '^version = "0.7.' "${ZEROCLAW_DIR}/Cargo.toml" 2>/dev/null \
   && ! grep -q '0\.7\.4' "${ZEROCLAW_DIR}/Cargo.toml" 2>/dev/null; then
    echo "WARNING: ${ZEROCLAW_DIR}/Cargo.toml does not mention version 0.7.x."
    echo "         The files in this patch were generated against v0.7.4 and may"
    echo "         not fit other versions. Continuing anyway..."
fi

# If the target is a git working tree, require it to be clean. This makes
# `git checkout -- .` a one-shot revert. For non-git trees (tarballs,
# zips), skip the check — there's no revert mechanism to protect.
if [ -e "${ZEROCLAW_DIR}/.git" ]; then
    if [ -n "$(git -C "${ZEROCLAW_DIR}" status --porcelain 2>/dev/null)" ]; then
        echo "ERROR: ${ZEROCLAW_DIR} has uncommitted changes."
        echo "       Refusing to overlay — you'd lose the ability to revert."
        echo ""
        echo "       Either commit/stash your changes first, or pass --force"
        echo "       (NOT IMPLEMENTED — please be deliberate here)."
        exit 1
    fi
fi

echo "============================================================"
echo "ZeroClaw Avatar Channel Patch — v0.7.4"
echo "============================================================"
echo "Target:  ${ZEROCLAW_DIR}"
echo "Source:  ${FILES_DIR}"
echo ""

if [ -e "${ZEROCLAW_DIR}/.git" ]; then
    echo "HEAD:    $(git -C "${ZEROCLAW_DIR}" rev-parse --short HEAD)"
    echo "Branch:  $(git -C "${ZEROCLAW_DIR}" branch --show-current 2>/dev/null || echo '(detached)')"
    echo ""
fi

# Show what's about to be overwritten.
echo "Files to overlay:"
(cd "${FILES_DIR}" && find . -type f) | sed 's|^\./|  |'
echo ""

# Use rsync if available (preserves permissions, atomic per-file), fall
# back to cp -R otherwise. Both produce the same end state.
if command -v rsync >/dev/null 2>&1; then
    rsync -a "${FILES_DIR}/" "${ZEROCLAW_DIR}/"
else
    # cp -R copies directory contents recursively; the trailing /. on
    # the source ensures the contents (not the dir itself) are copied.
    cp -R "${FILES_DIR}/." "${ZEROCLAW_DIR}/"
fi

echo "============================================================"
echo "Patch applied successfully."
echo ""
echo "Next steps:"
echo "  1. cd ${ZEROCLAW_DIR}"
echo "  2. docker compose up -d --build         # recommended"
echo "     OR: cargo build --workspace"
echo "  3. docker exec zeroclaw sed -i 's/require_pairing = false/require_pairing = true/' \\"
echo "       /zeroclaw-data/.zeroclaw/config.toml"
echo "  4. docker exec zeroclaw zeroclaw config set providers.fallback openai"
echo "  5. docker restart zeroclaw"
echo "  6. Generate the bearer token (see README.md → Get the bearer token)"
echo "  7. Update playground/AGENTS.md with the {speech, content}"
echo "     response-format guidance (see README.md)"
echo ""
if [ -e "${ZEROCLAW_DIR}/.git" ]; then
    echo "To revert:"
    echo "  git -C ${ZEROCLAW_DIR} checkout -- ."
fi
echo "============================================================"
