#!/bin/bash
# ============================================================
# OpenClaw Avatar SSE Patch — v2026.5.6
# ============================================================
# Overlays the patched files from ./files/ onto an OpenClaw v2026.5.6
# checkout. Adds the nyxclaw avatar SSE channel
# (/v1/chat/completions/avatar) plus the lazy-loaded module wiring
# in src/gateway/server-http.ts.
#
# Usage:
#   ./patch.sh /path/to/openclaw-v2026.5.6
#
# What it does:
#   1. Sanity-check the target is OpenClaw v2026.5.6 source
#   2. If the target is a git checkout, refuse if there are uncommitted
#      changes (so a future revert via `git checkout -- .` is clean)
#   3. Copy every file from ./files/ into the target, preserving paths
#   4. Print next-step build/run commands
#
# Reverting:
#   - If the target is a git checkout: `git -C <target> checkout -- .`
#     resets every modified file to its upstream v2026.5.6 state.
#   - Otherwise: re-extract the upstream source.
#
# NOTE: After patching you must add the {speech, content} response-format
# guidance to your workspace AGENTS.md. See README.md.
# ============================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path-to-openclaw-v2026.5.6>"
    echo ""
    echo "Example: $0 /home/user/openclaw-v2026.5.6"
    exit 1
fi

OPENCLAW_DIR="$1"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
FILES_DIR="${PATCH_DIR}/files"

if [ ! -d "${FILES_DIR}" ]; then
    echo "ERROR: ${FILES_DIR} does not exist — patch directory is incomplete."
    exit 1
fi

if [ ! -f "${OPENCLAW_DIR}/package.json" ]; then
    echo "ERROR: ${OPENCLAW_DIR}/package.json not found. Is this an OpenClaw checkout?"
    exit 1
fi

if [ ! -f "${OPENCLAW_DIR}/src/gateway/server-http.ts" ]; then
    echo "ERROR: ${OPENCLAW_DIR}/src/gateway/server-http.ts not found."
    exit 1
fi

# Light version sanity-check — bails out if package.json looks unrelated.
# Doesn't try to be too clever; the real safety is the git-clean check below.
if ! grep -q '"version": "2026\.5\.' "${OPENCLAW_DIR}/package.json" 2>/dev/null; then
    echo "WARNING: ${OPENCLAW_DIR}/package.json does not mention version 2026.5.x."
    echo "         The files in this patch were generated against v2026.5.6 and may"
    echo "         not fit other versions. Continuing anyway..."
fi

# If the target is a git working tree, require it to be clean. This makes
# `git checkout -- .` a one-shot revert. For non-git trees (tarballs,
# zips), skip the check — there's no revert mechanism to protect.
if [ -e "${OPENCLAW_DIR}/.git" ]; then
    if [ -n "$(git -C "${OPENCLAW_DIR}" status --porcelain 2>/dev/null)" ]; then
        echo "ERROR: ${OPENCLAW_DIR} has uncommitted changes."
        echo "       Refusing to overlay — you'd lose the ability to revert."
        echo ""
        echo "       Either commit/stash your changes first, or pass --force"
        echo "       (NOT IMPLEMENTED — please be deliberate here)."
        exit 1
    fi
fi

echo "============================================================"
echo "OpenClaw Avatar SSE Patch — v2026.5.6"
echo "============================================================"
echo "Target:  ${OPENCLAW_DIR}"
echo "Source:  ${FILES_DIR}"
echo ""

if [ -e "${OPENCLAW_DIR}/.git" ]; then
    echo "HEAD:    $(git -C "${OPENCLAW_DIR}" rev-parse --short HEAD)"
    echo "Branch:  $(git -C "${OPENCLAW_DIR}" branch --show-current 2>/dev/null || echo '(detached)')"
    echo ""
fi

# Show what's about to be overwritten.
echo "Files to overlay:"
(cd "${FILES_DIR}" && find . -type f) | sed 's|^\./|  |'
echo ""

# Use rsync if available (preserves permissions, atomic per-file), fall
# back to cp -R otherwise. Both produce the same end state.
if command -v rsync >/dev/null 2>&1; then
    rsync -a "${FILES_DIR}/" "${OPENCLAW_DIR}/"
else
    # cp -R copies directory contents recursively; the trailing /. on
    # the source ensures the contents (not the dir itself) are copied.
    cp -R "${FILES_DIR}/." "${OPENCLAW_DIR}/"
fi

echo "============================================================"
echo "Patch applied successfully."
echo ""
echo "Next steps:"
echo "  1. cd ${OPENCLAW_DIR}"
echo "  2. npm install"
echo "  3. npm run build"
echo "  4. npm start -- gateway --bind lan --port 18789"
echo "  5. Set OPENCLAW_GATEWAY_TOKEN in your env (or docker-compose.yml)"
echo "  6. Add {speech, content} response-format guidance to your"
echo "     workspace AGENTS.md (see README.md)"
echo ""
if [ -e "${OPENCLAW_DIR}/.git" ]; then
    echo "To revert:"
    echo "  git -C ${OPENCLAW_DIR} checkout -- ."
fi
echo "============================================================"
