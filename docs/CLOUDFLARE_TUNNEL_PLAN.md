# Cloudflare Tunnel Integration Plan

Last updated: 2026-03-22

## Goal

Enable `wss://` connections to self-hosted nyxclaw instances without requiring users to configure port forwarding, TLS certificates, or DNS. A mobile app should be able to scan a QR code and connect securely from anywhere.

## Current State

- Mobile app connects via `ws://<ip>:<port>/ws` (plaintext)
- Android requires `cleartextTrafficPermitted="true"` workaround
- iOS ATS blocks cleartext by default
- Not acceptable for app store distribution

## Target State

- nyxclaw auto-provisions a Cloudflare Tunnel on first boot
- Mobile app connects via `wss://<device-id>.nyxclaw.ai/ws`
- TLS terminated at Cloudflare edge — nyxclaw stays plain HTTP internally
- No port forwarding, no certs, no DNS config for the user
- Works behind CGNAT, firewalls, any ISP

## Architecture

```
Mobile App
    │
    │  wss://<device-id>.d.nyxclaw.ai/ws
    ▼
Cloudflare Edge (TLS termination)
    │
    │  Cloudflare Tunnel (encrypted, outbound-only)
    ▼
cloudflared (sidecar container or local daemon)
    │
    │  http://localhost:8080 (plain HTTP)
    ▼
nyxclaw server
    │
    │  Auth: device challenge + token verification
    ▼
Session established
```

### What each layer does

| Layer | Responsibility | Auth? |
|-------|---------------|-------|
| Cloudflare Edge | TLS termination, DDoS protection | No — passes traffic through |
| cloudflared | Routes tunnel traffic to local nyxclaw | No — just plumbing |
| nyxclaw | Device auth (Ed25519 challenge + token) | Yes — all auth happens here |

Cloudflare is invisible infrastructure. It does NOT authenticate clients.

## Provisioning Flow

### First boot (automatic)

```
1. nyxclaw starts
2. Checks data/device_id — not found, generates UUID (e.g. "a3f7b2c1")
3. Checks data/tunnel.json — not found, needs provisioning
4. Calls POST api.nyxclaw.ai/provision
   Request:  { "device_id": "a3f7b2c1" }
   Response: { "tunnel_token": "eyJ...", "hostname": "a3f7b2c1.d.nyxclaw.ai" }
5. Stores tunnel_token + hostname in data/tunnel.json
6. cloudflared reads token, connects to Cloudflare
7. wss://a3f7b2c1.d.nyxclaw.ai is live
8. QR code / setup code uses wss://a3f7b2c1.d.nyxclaw.ai/ws as the URL
```

### Subsequent boots

```
1. nyxclaw starts
2. Reads data/device_id → "a3f7b2c1"
3. Reads data/tunnel.json → has tunnel_token + hostname
4. cloudflared starts with stored token
5. Tunnel reconnects in ~5 seconds
6. QR code uses stored hostname
```

### Re-provisioning

If `data/tunnel.json` is deleted, nyxclaw re-provisions on next boot. The provisioning API is idempotent — same device_id returns the same tunnel.

### Shutdown / uninstall

The user does NOT need anything running for cleanup to happen. The provisioning Worker (`api.nyxclaw.ai`) is Myned's centrally-hosted service, always running on Cloudflare's edge.

**User stops containers / turns off machine:**
- Tunnel goes inactive (`conns_inactive_at` timestamp set by Cloudflare)
- Worker's daily cleanup deletes tunnels inactive for 5+ days, freeing the DNS slot
- If user starts again within 5 days: cloudflared reconnects, no re-provisioning needed
- If user starts again after 5+ days: nyxclaw detects stale token (section 1.7), re-provisions automatically

**User uninstalls everything (containers, code, data — all gone):**
- Tunnel orphaned — Worker cleans it up after 5 days
- No user action needed

**User deletes `data/` but keeps the code:**
- New device_id generated on next boot
- Old tunnel orphaned → cleaned up after 5 days
- New tunnel provisioned with new device_id

## Implementation Plan

### Phase 1: nyxclaw server changes

#### 1.1 Device ID generation

**File:** `src/services/tunnel_service.py` (new)

- On first boot, generate a short ID: 8 hex chars from `uuid4` (e.g. `a3f7b2c1`)
- Save to `data/device_id` (plain text file)
- On subsequent boots, read from file
- Never changes unless user deletes the file

#### 1.2 Provisioning client

**File:** `src/services/tunnel_service.py`

- `async def provision_tunnel(device_id: str) -> TunnelConfig`
- Calls `POST {PROVISIONING_API_URL}/provision` with `{"device_id": device_id}`
- Receives `{"tunnel_token": "...", "hostname": "..."}`
- Stores in `data/tunnel.json`
- Retries with backoff on failure (network might not be ready at boot)
- Timeout: 30 seconds

#### 1.3 Settings updates

**File:** `src/core/settings.py`

New fields:
```python
provisioning_api_url: str = "https://api.nyxclaw.ai"
device_id_path: str = "./data/device_id"
tunnel_config_path: str = "./data/tunnel.json"
```

#### 1.4 Startup integration

**File:** `src/main.py`

On app startup (lifespan):
1. Load or generate device_id
2. Load or provision tunnel config
3. Derive `auth_setup_code_url` from tunnel hostname: `wss://{hostname}/ws`
4. Generate QR code with the wss:// URL

#### 1.5 QR code / setup code update

**File:** `src/auth/setup_code_service.py`

- `auth_setup_code_url` is already used to build the QR payload
- No code change needed IF we set it from the tunnel hostname at startup
- The setup code JSON will now contain `wss://a3f7b2c1.d.nyxclaw.ai/ws` instead of `ws://192.168.1.50:8081/ws`

#### 1.6 Docker Compose

**File:** `docker-compose.yml`

Add tunnel sidecar:
```yaml
  tunnel:
    image: cloudflare/cloudflared:latest
    container_name: nyxclaw-tunnel
    entrypoint: ["/bin/sh", "-c"]
    command:
      - |
        echo "Waiting for tunnel token..."
        while [ ! -f /data/tunnel_token ]; do sleep 2; done
        echo "Token found, starting tunnel"
        exec cloudflared tunnel run --token "$(cat /data/tunnel_token)"
    volumes:
      - ./data:/data:ro
    depends_on:
      - server
    restart: unless-stopped
```

**Token delivery:** nyxclaw writes `data/tunnel_token` as a plain text file (just the raw token string, not JSON). cloudflared reads it via the entrypoint script above.

**First-boot ordering:** On first boot, nyxclaw provisions the tunnel and writes `data/tunnel_token`. The tunnel sidecar polls for the file every 2 seconds. Once the file appears, cloudflared starts. On subsequent boots, the file already exists so cloudflared starts immediately. `restart: unless-stopped` ensures the sidecar recovers from any transient failures.

#### 1.7 Stale token detection

If the server was offline for 5+ days, the Worker cleanup may have deleted the tunnel. On next boot:

1. cloudflared starts with the stored token but fails to connect (tunnel no longer exists)
2. nyxclaw detects this via a health check: `GET api.nyxclaw.ai/provision/{device_id}/status`
3. If the tunnel no longer exists, nyxclaw deletes `data/tunnel.json` + `data/tunnel_token` and re-provisions
4. cloudflared sidecar detects the new token file and restarts

The health check runs once at startup (after loading the stored config) and periodically (every 6 hours) to catch edge cases.

#### 1.8 Graceful fallback

If provisioning fails (no internet, API down), nyxclaw should still start:
- Log a warning: "Tunnel not available — server accessible on local network only"
- QR code falls back to `ws://<local-ip>:<port>/ws`
- Retry provisioning on a timer (every 5 minutes) until successful
- Once provisioned, update the QR code URL to `wss://` automatically

### Phase 2: Provisioning API (api.nyxclaw.ai)

#### 2.1 Technology

Cloudflare Worker (runs on Cloudflare's edge, free tier, zero infra to manage). The Worker calls the Cloudflare API to create tunnels and DNS records using a service API token.

#### 2.2 Endpoints

**`POST /provision`**

```
Request:
{
  "device_id": "a3f7b2c1"
}

Response (201 Created):
{
  "tunnel_token": "eyJ...",
  "hostname": "a3f7b2c1.d.nyxclaw.ai",
  "tunnel_id": "uuid-of-tunnel"
}

Response (200 OK — already provisioned):
{
  "tunnel_token": "eyJ...",
  "hostname": "a3f7b2c1.d.nyxclaw.ai",
  "tunnel_id": "uuid-of-tunnel"
}
```

Idempotent — same device_id always returns the same tunnel.

**`DELETE /provision/:device_id`**

Deletes the tunnel and DNS record. For decommissioning a server.

**`GET /provision/:device_id/status`**

Returns tunnel status (active/inactive). For health checks.

#### 2.3 What the Worker does internally

1. Receives device_id
2. Calls Cloudflare API: Create Tunnel (name: `nyxclaw-{device_id}`)
3. Calls Cloudflare API: Create CNAME record (`{device_id}.d.nyxclaw.ai` → tunnel UUID)
4. Calls Cloudflare API: Set tunnel ingress config (route to `http://server:8080`)
5. Returns the tunnel token

#### 2.4 Security & Abuse Prevention

- The Worker's Cloudflare API token is stored as a Worker secret (never exposed)
- Device IDs are validated (alphanumeric + hyphens only, 4-32 chars)
- **Rate limit: 5 provisions per IP per day** — enough for testing, blocks bulk abuse
- No API keys or activation codes required — open provisioning keeps onboarding frictionless
- Abuse self-corrects via stale tunnel cleanup (see 2.6)

#### 2.5 Cloudflare API token permissions (for the Worker)

- Account : Cloudflare Tunnel : Edit
- Zone : DNS : Edit (scoped to nyxclaw.ai zone)

#### 2.6 Stale Tunnel Cleanup

Cloudflare does NOT auto-delete inactive tunnels. The provisioning Worker runs a **daily scheduled cleanup**:

1. List all tunnels via Cloudflare API
2. Find tunnels where `conns_inactive_at` is older than **5 days**
3. Delete the tunnel + its CNAME record
4. This is safe because provisioning is idempotent — if a user turns their server back on, it re-provisions automatically

This keeps the pool clean and prevents hitting the 1,000 tunnel / 200 DNS record limits.

#### 2.7 Cloudflare Free Tier Limits

| Resource | Limit | Impact |
|----------|-------|--------|
| Tunnels per account | 1,000 | Comfortable headroom |
| DNS records (free, zone created after Sept 2024) | 200 | **Real bottleneck** — 1 CNAME per device, ~200 active devices max |
| Pro plan | 3,500 DNS records | $20/mo upgrade when needed |

With 5-day cleanup, you'd need 200+ simultaneously active servers to hit the free tier DNS limit. Upgrade to Pro ($20/mo) when user base approaches ~150 active devices.

### Phase 3: Install scripts

#### 3.1 install.sh (Linux/macOS)

```
1. Check prerequisites (curl, python3.10+)
2. Install uv (if not present)
3. Install cloudflared (if not present)
4. Clone/download nyxclaw
5. uv sync
6. Download Wav2Arkit model
7. Copy .env.example → .env
8. Run first-boot provisioning (python -c "from services.tunnel_service import ...")
9. Install systemd service (Linux) or launchd plist (macOS)
   - nyxclaw.service: runs uv run python src/main.py
   - nyxclaw-tunnel.service: runs cloudflared tunnel run --token <token>
   - tunnel service BindsTo nyxclaw (stops together)
10. Start services
11. Print QR code to terminal
```

#### 3.2 install.ps1 (Windows)

Same flow but:
- Uses `winget` for cloudflared
- Uses `nssm` or `New-Service` for Windows services
- Adjusts paths for Windows

### Phase 4: Mobile app changes

#### 4.1 Required changes (blocking)

| Change | Description |
|--------|-------------|
| Accept `wss://` URLs | The setup code will now contain `wss://` URLs. The app must support both `ws://` and `wss://` — most WebSocket libraries handle this automatically. |
| Remove cleartext workaround | Android: revert `cleartextTrafficPermitted="true"` in `network_security_config.xml`. iOS: remove any ATS exceptions. |

#### 4.2 Recommended changes (non-blocking)

| Change | Description |
|--------|-------------|
| Connection status indicator | Show "Connected via nyxclaw.ai" or "Connected locally" so user knows which path is active |
| Fallback to local | If `wss://` fails (tunnel down), attempt `ws://<local-ip>` as fallback (requires mDNS or cached IP) |
| Certificate pinning | Optional: pin the Cloudflare edge cert for extra security (prevents MITM even with compromised CA) |

#### 4.3 No changes needed

| Area | Why |
|------|-----|
| Auth flow | Same device challenge + token — transport is transparent |
| WebSocket protocol | Same message types, same binary format |
| Audio format | Same PCM16 24kHz |
| QR scanning | Same JSON payload format, just URL field changes from `ws://` to `wss://` |

### Phase 5: Testing

#### 5.1 Provisioning

- [ ] First boot generates device_id and provisions tunnel
- [ ] Subsequent boots reuse stored tunnel config
- [ ] Deleting `data/tunnel.json` triggers re-provisioning
- [ ] Provisioning API is idempotent (same device_id = same tunnel)
- [ ] Provisioning failure degrades gracefully (local-only mode)
- [ ] Rate limiting on provisioning API works

#### 5.2 Tunnel connectivity

- [ ] cloudflared connects and stays connected
- [ ] WebSocket traffic passes through tunnel correctly
- [ ] Audio streaming works over tunnel (latency acceptable)
- [ ] Barge-in works over tunnel (cancel propagation)
- [ ] Tunnel reconnects after network interruption
- [ ] Tunnel survives server restart

#### 5.3 Mobile app

- [ ] QR code with `wss://` URL scans and parses correctly
- [ ] App connects via `wss://` and authenticates
- [ ] Audio + blendshapes stream correctly over `wss://`
- [ ] Rich content messages arrive
- [ ] Cleartext workaround removed, app still works

#### 5.4 Install scripts

- [ ] install.sh works on fresh Ubuntu 22.04
- [ ] install.sh works on fresh macOS
- [ ] install.ps1 works on Windows 11
- [ ] Services start on boot
- [ ] Services restart on crash

## File Changes Summary

| File | Change |
|------|--------|
| `src/services/tunnel_service.py` | **NEW** — device ID generation + provisioning client |
| `src/core/settings.py` | Add tunnel/provisioning settings |
| `src/main.py` | Call tunnel provisioning on startup |
| `docker-compose.yml` | Add tunnel sidecar service |
| `.env.example` | Add provisioning settings |
| `install.sh` | **NEW** — Linux/macOS install script |
| `install.ps1` | **NEW** — Windows install script |

## Resolved Questions

1. **Domain:** `nyxclaw.ai` — device subdomains under `<id>.d.nyxclaw.ai`
2. **Abuse prevention:** Open provisioning (no keys), rate limited to 5 per IP per day. Abuse self-corrects via 5-day stale cleanup.
3. **Tunnel limits:** 1,000 tunnels but only 200 DNS records on free tier (zone registered 2026-03-22). DNS is the bottleneck. 5-day cleanup keeps pool clean. Upgrade to Pro ($20/mo, 3,500 records) at ~150 active devices.
4. **Decommissioning:** Automated — 5-day inactive cleanup via scheduled Worker job. Install script can also call `DELETE /provision/:device_id` on uninstall.

## Open Questions

1. **Monitoring:** How to monitor tunnel health across all provisioned instances? Cloudflare dashboard shows per-tunnel status. Consider a simple status page or health endpoint on the Worker.
2. **Uninstall cleanup:** Should install scripts call the deprovision endpoint on uninstall? Or just let the 5-day cleanup handle it?
