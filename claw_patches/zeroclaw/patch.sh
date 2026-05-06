#!/bin/bash
# ============================================================
# ZeroClaw Avatar Channel Patch — v0.7.4
# ============================================================
# Applies the nyxclaw avatar channel patch to a ZeroClaw v0.7.4
# checkout via `git apply`.
#
# Usage:
#   ./patch.sh /path/to/zeroclaw-v0.7.4
#
# What it does:
#   - Verifies the target is a clean v0.7.4 checkout (commit 78fb0a6)
#   - Runs `git apply --check` to dry-run the patch
#   - Applies the patch with `git apply`
#   - Reports build/test commands to run next
#
# What the patch contains:
#   1. response_format threading — ChatRequest gains an optional
#      response_format field, plumbed through Agent::turn_streamed,
#      OpenAI native, Anthropic, Reliable, Router, OpenRouter.
#   2. OpenAI native SSE streaming — stream_chat() implementation
#      with per-delta tool-call accumulation and structured-output
#      compatibility.
#   3. SystemPromptBuilder::without_section() — lets the avatar strip
#      DateTimeSection so OpenAI's automatic prompt cache stays warm.
#   4. nyxclaw avatar channel — new /ws/avatar WebSocket endpoint
#      (zeroclaw-gateway::nyxclaw) emitting structured speech_chunk +
#      rich_content + tool_call/tool_result + filler frames with
#      barge-in support.
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
PATCH_FILE="${PATCH_DIR}/zeroclaw-v0.7.4-nyxclaw.patch"
EXPECTED_BASE="78fb0a6"  # zeroclaw v0.7.4 release tag commit

if [ ! -f "${PATCH_FILE}" ]; then
    echo "ERROR: patch file not found: ${PATCH_FILE}"
    exit 1
fi

if [ ! -f "${ZEROCLAW_DIR}/Cargo.toml" ]; then
    echo "ERROR: ${ZEROCLAW_DIR}/Cargo.toml not found. Is this a ZeroClaw checkout?"
    exit 1
fi

if [ ! -e "${ZEROCLAW_DIR}/.git" ]; then
    # `.git` is a directory in regular checkouts and a file in `git worktree`s.
    echo "ERROR: ${ZEROCLAW_DIR} is not a git repository. The patch requires git apply."
    echo "       Clone with: git clone https://github.com/zeroclaw-labs/zeroclaw -b v0.7.4 ${ZEROCLAW_DIR}"
    exit 1
fi

echo "============================================================"
echo "ZeroClaw Avatar Channel Patch — v0.7.4"
echo "============================================================"
echo "Target:  ${ZEROCLAW_DIR}"
echo "Patch:   ${PATCH_FILE}"
echo ""

# Confirm we're on (or at least contain) the v0.7.4 release commit
HEAD_COMMIT="$(git -C "${ZEROCLAW_DIR}" rev-parse --short HEAD)"
echo "HEAD:    ${HEAD_COMMIT}"

if ! git -C "${ZEROCLAW_DIR}" merge-base --is-ancestor "${EXPECTED_BASE}" HEAD 2>/dev/null; then
    echo ""
    echo "WARNING: ${EXPECTED_BASE} (v0.7.4 release tag) is not an ancestor of HEAD."
    echo "         The patch was generated against v0.7.4 and may not apply cleanly."
    echo ""
    read -r -p "Continue anyway? [y/N] " ans
    if [[ "${ans}" != "y" && "${ans}" != "Y" ]]; then
        exit 1
    fi
fi

# Dry-run first so we fail loud before any tree mutation
echo ""
echo "Dry-running patch (git apply --check)..."
if ! git -C "${ZEROCLAW_DIR}" apply --check "${PATCH_FILE}"; then
    echo ""
    echo "ERROR: patch does not apply cleanly. Aborting before any changes."
    echo "       Inspect ${PATCH_FILE} and resolve conflicts manually, or"
    echo "       reset the target to a clean v0.7.4 checkout."
    exit 1
fi
echo "  OK — patch applies cleanly."

echo ""
echo "Applying patch..."
git -C "${ZEROCLAW_DIR}" apply "${PATCH_FILE}"
echo "  OK — patch applied."

echo ""
echo "============================================================"
echo "Patch applied successfully."
echo ""
echo "Next steps:"
echo "  1. cd ${ZEROCLAW_DIR}"
echo "  2. cargo build --workspace"
echo "  3. cargo test -p zeroclaw-gateway --lib nyxclaw"
echo "  4. Update playground/AGENTS.md with the {speech, content}"
echo "     response-format guidance (see ${PATCH_DIR}/README.md)."
echo "  5. Start the gateway and connect nyxclaw to /ws/avatar."
echo ""
echo "To revert:"
echo "  git -C ${ZEROCLAW_DIR} apply -R ${PATCH_FILE}"
echo "============================================================"
