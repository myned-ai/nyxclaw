#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# OpenClaw Avatar SSE Patch
# ============================================================
# Adds /v1/chat/completions/avatar endpoint for nyxclaw integration.
#
# Usage:
#   ./patch.sh /path/to/openclaw-v2026.3.13
# ============================================================

OPENCLAW_DIR="${1:?Usage: ./patch.sh /path/to/openclaw}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKUP_DIR="${OPENCLAW_DIR}/.nyxclaw-patch-backup"

echo "============================================================"
echo "OpenClaw Avatar SSE Patch"
echo "============================================================"
echo "Target:  ${OPENCLAW_DIR}"
echo "Patches: ${SCRIPT_DIR}"
echo ""

# ── Validate ───────────────────────────────────────────────────
if [ ! -f "${OPENCLAW_DIR}/src/gateway/openai-http.ts" ]; then
  echo "ERROR: ${OPENCLAW_DIR}/src/gateway/openai-http.ts not found."
  echo "       Is this a valid OpenClaw source directory?"
  exit 1
fi

if [ ! -f "${OPENCLAW_DIR}/src/gateway/server-http.ts" ]; then
  echo "ERROR: ${OPENCLAW_DIR}/src/gateway/server-http.ts not found."
  exit 1
fi

echo "Backups: ${BACKUP_DIR}"
echo ""

# ── Step 1: Copy avatar-http.ts ────────────────────────────────
echo "Step 1: Installing avatar-http.ts..."
echo ""

mkdir -p "${BACKUP_DIR}/src/gateway"

# Backup server-http.ts (we'll inject into it)
if [ ! -f "${BACKUP_DIR}/src/gateway/server-http.ts" ]; then
  cp "${OPENCLAW_DIR}/src/gateway/server-http.ts" "${BACKUP_DIR}/src/gateway/server-http.ts"
  echo "  BACKUP src/gateway/server-http.ts"
fi

# Copy new avatar handler
cp "${SCRIPT_DIR}/src/gateway/avatar-http.ts" "${OPENCLAW_DIR}/src/gateway/avatar-http.ts"
echo "  NEW    src/gateway/avatar-http.ts"

echo ""

# ── Step 2: Inject import + route into server-http.ts ──────────
echo "Step 2: Registering avatar route in server-http.ts..."
echo ""

SERVER_HTTP="${OPENCLAW_DIR}/src/gateway/server-http.ts"

# 2a: Add import (after the openai-http import)
if grep -q 'handleAvatarHttpRequest' "${SERVER_HTTP}" 2>/dev/null; then
  echo "  SKIP  import (already present)"
else
  # Find the openai-http import line and add our import after it
  sed -i.bak '/import { handleOpenAiHttpRequest } from "\.\/openai-http\.js";/a\
import { handleAvatarHttpRequest } from "./avatar-http.js";' "${SERVER_HTTP}"
  rm -f "${SERVER_HTTP}.bak"
  echo "  INJECT import { handleAvatarHttpRequest } from \"./avatar-http.js\""
fi

# 2b: Add route stage (before the openai stage)
if grep -q '"avatar"' "${SERVER_HTTP}" 2>/dev/null; then
  echo "  SKIP  route stage (already present)"
else
  # Find the openai stage push and insert avatar stage before it
  # We match: if (openAiChatCompletionsEnabled) {
  # And insert the avatar stage right before it
  AVATAR_STAGE='      if (openAiChatCompletionsEnabled) {\
        requestStages.push({\
          name: "avatar",\
          run: () =>\
            handleAvatarHttpRequest(req, res, {\
              auth: resolvedAuth,\
              config: openAiChatCompletionsConfig,\
              trustedProxies,\
              allowRealIpFallback,\
              rateLimiter,\
            }),\
        });\
      }'

  # Use awk to inject before the first `if (openAiChatCompletionsEnabled)` in the requestStages section
  awk -v stage="${AVATAR_STAGE}" '
    /if \(openAiChatCompletionsEnabled\)/ && !done {
      print stage
      done = 1
    }
    { print }
  ' "${SERVER_HTTP}" > "${SERVER_HTTP}.tmp"
  mv "${SERVER_HTTP}.tmp" "${SERVER_HTTP}"
  echo "  INJECT route stage 'avatar' (before 'openai' stage)"
fi

echo ""

# ── Verify ─────────────────────────────────────────────────────
echo "Verifying..."

ERRORS=0

if [ ! -f "${OPENCLAW_DIR}/src/gateway/avatar-http.ts" ]; then
  echo "  FAIL  avatar-http.ts not found"
  ERRORS=$((ERRORS + 1))
else
  echo "  OK    avatar-http.ts"
fi

if ! grep -q 'handleAvatarHttpRequest' "${SERVER_HTTP}"; then
  echo "  FAIL  avatar import not in server-http.ts"
  ERRORS=$((ERRORS + 1))
else
  echo "  OK    avatar import in server-http.ts"
fi

if ! grep -q '"avatar"' "${SERVER_HTTP}"; then
  echo "  FAIL  avatar route stage not in server-http.ts"
  ERRORS=$((ERRORS + 1))
else
  echo "  OK    avatar route stage in server-http.ts"
fi

if [ "${ERRORS}" -gt 0 ]; then
  echo ""
  echo "ERROR: ${ERRORS} verification(s) failed."
  exit 1
fi

echo "  All patches verified."
echo ""
echo "============================================================"
echo "Patch applied successfully!"
echo ""
echo "Next steps:"
echo "  1. cd ${OPENCLAW_DIR}"
echo "  2. npm run build   (or: pnpm build)"
echo "  3. npm start -- gateway --bind lan"
echo "  4. Connect nyxclaw to http://<host>:<port>/v1/chat/completions/avatar"
echo "============================================================"
