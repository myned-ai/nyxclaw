#!/usr/bin/env bash
# ============================================================
# NyxClaw Install Script (Linux / macOS)
# ============================================================
# Installs nyxclaw + cloudflared, provisions a tunnel,
# and sets up system services so everything starts on boot.
#
# Usage:
#   curl -sSL https://nyxclaw.ai/install.sh | bash
#   # or:
#   ./install.sh
# ============================================================

set -euo pipefail

INSTALL_DIR="${NYXCLAW_DIR:-$HOME/nyxclaw}"
REPO_URL="https://github.com/myned-ai/nyxclaw.git"
PROVISIONING_API="https://nyxclaw-provisioner.nyxclaw.workers.dev"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[nyxclaw]${NC} $1"; }
ok()    { echo -e "${GREEN}[nyxclaw]${NC} $1"; }
warn()  { echo -e "${YELLOW}[nyxclaw]${NC} $1"; }
fail()  { echo -e "${RED}[nyxclaw]${NC} $1"; exit 1; }

# ── OS detection ─────────────────────────────────────────────

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux*)  PLATFORM="linux" ;;
    Darwin*) PLATFORM="macos" ;;
    *)       fail "Unsupported OS: $OS. Use install.ps1 for Windows." ;;
esac

info "Platform: $PLATFORM ($ARCH)"

# ── Prerequisites ────────────────────────────────────────────

check_cmd() {
    command -v "$1" &>/dev/null
}

# Python 3.10+
if check_cmd python3; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
        fail "Python 3.10+ required (found $PY_VERSION). Install from https://python.org"
    fi
    ok "Python $PY_VERSION"
else
    fail "Python 3 not found. Install from https://python.org"
fi

# ── Install uv ───────────────────────────────────────────────

if check_cmd uv; then
    ok "uv $(uv --version 2>/dev/null | head -1)"
else
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    check_cmd uv || fail "uv installation failed"
    ok "uv installed"
fi

# ── Install cloudflared ──────────────────────────────────────

if check_cmd cloudflared; then
    ok "cloudflared $(cloudflared --version 2>&1 | head -1)"
else
    info "Installing cloudflared..."
    if [ "$PLATFORM" = "macos" ]; then
        if check_cmd brew; then
            brew install cloudflared
        else
            fail "Install Homebrew first (https://brew.sh) or install cloudflared manually"
        fi
    else
        # Linux
        if [ "$ARCH" = "x86_64" ]; then
            CF_ARCH="amd64"
        elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
            CF_ARCH="arm64"
        else
            fail "Unsupported architecture: $ARCH"
        fi
        curl -L "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CF_ARCH}" -o /usr/local/bin/cloudflared 2>/dev/null \
            || sudo curl -L "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CF_ARCH}" -o /usr/local/bin/cloudflared
        chmod +x /usr/local/bin/cloudflared 2>/dev/null || sudo chmod +x /usr/local/bin/cloudflared
    fi
    check_cmd cloudflared || fail "cloudflared installation failed"
    ok "cloudflared installed"
fi

# ── Clone / update nyxclaw ───────────────────────────────────

if [ -d "$INSTALL_DIR/.git" ]; then
    info "Updating nyxclaw in $INSTALL_DIR..."
    cd "$INSTALL_DIR"
    git pull --ff-only 2>/dev/null || warn "Git pull failed — using existing code"
else
    info "Cloning nyxclaw to $INSTALL_DIR..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

ok "nyxclaw at $INSTALL_DIR"

# ── Install Python dependencies ──────────────────────────────

info "Installing Python dependencies..."
uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev
ok "Dependencies installed"

# ── Download Wav2Arkit model ─────────────────────────────────

MODEL_DIR="$INSTALL_DIR/pretrained_models/wav2arkit"
if [ -f "$MODEL_DIR/wav2arkit_cpu.onnx" ]; then
    ok "Wav2Arkit model already downloaded"
else
    info "Downloading Wav2Arkit model..."
    mkdir -p "$MODEL_DIR"
    uv run --with "huggingface_hub[cli]" huggingface-cli download myned-ai/wav2arkit_cpu --local-dir "$MODEL_DIR"
    ok "Wav2Arkit model downloaded"
fi

# ── Create .env if needed ────────────────────────────────────

if [ ! -f "$INSTALL_DIR/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    chmod 600 "$INSTALL_DIR/.env"
    warn "Created .env from template — edit it with your backend settings (BASE_URL, AUTH_TOKEN, etc.)"
else
    ok ".env already exists"
fi

# ── Provision Cloudflare Tunnel ──────────────────────────────

info "Provisioning Cloudflare Tunnel..."
cd "$INSTALL_DIR"
PROVISIONING_API_URL="$PROVISIONING_API" uv run python -c "
import sys, asyncio
sys.path.insert(0, 'src')
from services.tunnel_service import ensure_tunnel
async def run():
    config = await ensure_tunnel(
        device_id_path='./data/device_id',
        tunnel_config_path='./data/tunnel.json',
        provisioning_api_url='$PROVISIONING_API',
    )
    if config:
        print(f'TUNNEL_OK:{config.hostname}')
    else:
        print('TUNNEL_FAIL')
result = asyncio.run(run())
" 2>/dev/null | while read -r line; do
    case "$line" in
        TUNNEL_OK:*)
            HOSTNAME="${line#TUNNEL_OK:}"
            ok "Tunnel provisioned: wss://$HOSTNAME/ws"
            ;;
        TUNNEL_FAIL)
            warn "Tunnel provisioning failed — server will be local-only"
            ;;
    esac
done

# ── Install system services ──────────────────────────────────

TUNNEL_TOKEN_FILE="$INSTALL_DIR/data/tunnel_token"

if [ "$PLATFORM" = "linux" ]; then
    info "Installing systemd services..."

    # nyxclaw service
    sudo tee /etc/systemd/system/nyxclaw.service > /dev/null << SVCEOF
[Unit]
Description=NyxClaw Voice-to-Avatar Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$(which uv) run --frozen --no-dev python src/main.py
Restart=always
RestartSec=5
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin

[Install]
WantedBy=multi-user.target
SVCEOF

    # cloudflared tunnel service
    if [ -f "$TUNNEL_TOKEN_FILE" ]; then
        sudo tee /etc/systemd/system/nyxclaw-tunnel.service > /dev/null << TUNEOF
[Unit]
Description=NyxClaw Cloudflare Tunnel
After=network-online.target nyxclaw.service
Wants=network-online.target
BindsTo=nyxclaw.service

[Service]
Type=simple
ExecStart=$(which cloudflared) tunnel run --token-file $TUNNEL_TOKEN_FILE
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
TUNEOF
        sudo systemctl daemon-reload
        sudo systemctl enable nyxclaw nyxclaw-tunnel
        sudo systemctl start nyxclaw nyxclaw-tunnel
        ok "Services installed and started (nyxclaw + tunnel)"
    else
        sudo systemctl daemon-reload
        sudo systemctl enable nyxclaw
        sudo systemctl start nyxclaw
        ok "Service installed and started (nyxclaw only — no tunnel token)"
    fi

elif [ "$PLATFORM" = "macos" ]; then
    info "Installing launchd services..."

    PLIST_DIR="$HOME/Library/LaunchAgents"
    mkdir -p "$PLIST_DIR"

    # nyxclaw service
    cat > "$PLIST_DIR/ai.nyxclaw.server.plist" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.nyxclaw.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(which uv)</string>
        <string>run</string>
        <string>--frozen</string>
        <string>--no-dev</string>
        <string>python</string>
        <string>src/main.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$INSTALL_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$INSTALL_DIR/data/nyxclaw.log</string>
    <key>StandardErrorPath</key>
    <string>$INSTALL_DIR/data/nyxclaw.log</string>
</dict>
</plist>
PLISTEOF

    launchctl load "$PLIST_DIR/ai.nyxclaw.server.plist" 2>/dev/null || true
    ok "nyxclaw service installed"

    # cloudflared tunnel service
    if [ -f "$TUNNEL_TOKEN_FILE" ]; then
        cat > "$PLIST_DIR/ai.nyxclaw.tunnel.plist" << TUNPLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>ai.nyxclaw.tunnel</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(which cloudflared)</string>
        <string>tunnel</string>
        <string>run</string>
        <string>--token-file</string>
        <string>$TUNNEL_TOKEN_FILE</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$INSTALL_DIR/data/tunnel.log</string>
    <key>StandardErrorPath</key>
    <string>$INSTALL_DIR/data/tunnel.log</string>
</dict>
</plist>
TUNPLISTEOF

        launchctl load "$PLIST_DIR/ai.nyxclaw.tunnel.plist" 2>/dev/null || true
        ok "Tunnel service installed"
    fi
fi

# ── Done ─────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo -e "${GREEN}NyxClaw installed successfully!${NC}"
echo "============================================================"
echo ""
echo "  Install dir:  $INSTALL_DIR"

if [ -f "$INSTALL_DIR/data/tunnel.json" ]; then
    HOSTNAME=$(python3 -c "import json; print(json.load(open('$INSTALL_DIR/data/tunnel.json'))['hostname'])" 2>/dev/null || echo "unknown")
    echo "  Tunnel:       wss://$HOSTNAME/ws"
fi

echo ""
echo "  Next steps:"
echo "    1. Edit $INSTALL_DIR/.env with your backend settings"
echo "    2. Restart: $([ "$PLATFORM" = "linux" ] && echo "sudo systemctl restart nyxclaw" || echo "launchctl kickstart -k gui/$(id -u)/ai.nyxclaw.server")"
echo ""
echo "  Logs:"
if [ "$PLATFORM" = "linux" ]; then
    echo "    journalctl -u nyxclaw -f"
    echo "    journalctl -u nyxclaw-tunnel -f"
else
    echo "    tail -f $INSTALL_DIR/data/nyxclaw.log"
    echo "    tail -f $INSTALL_DIR/data/tunnel.log"
fi
echo ""
echo "  Uninstall:"
if [ "$PLATFORM" = "linux" ]; then
    echo "    sudo systemctl stop nyxclaw nyxclaw-tunnel"
    echo "    sudo systemctl disable nyxclaw nyxclaw-tunnel"
    echo "    sudo rm /etc/systemd/system/nyxclaw*.service"
    echo "    rm -rf $INSTALL_DIR"
else
    echo "    launchctl unload ~/Library/LaunchAgents/ai.nyxclaw.*.plist"
    echo "    rm ~/Library/LaunchAgents/ai.nyxclaw.*.plist"
    echo "    rm -rf $INSTALL_DIR"
fi
echo ""
