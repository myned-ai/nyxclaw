# Security Gap Analysis

Last updated: 2026-03-22

## Current Security Posture

NyxClaw uses a device challenge + token auth model designed for a single-user self-hosted server. The mobile app pairs via QR code (TOFU — Trust On First Use), then authenticates subsequent connections with a device token.

### What's in place

| Feature | Implementation | Status |
|---------|---------------|--------|
| Device pairing | QR code / setup code with bootstrap token | Working |
| Challenge-response auth | Ed25519 signature verification | Working |
| Token storage | SHA-256 hashed in `auth_store.json` | Working |
| Nonce replay prevention | Single-use nonces with 30s TTL, consumed under lock | Working |
| Rate limiting | Token bucket per-session and per-domain on auth endpoints | Working |
| Transport encryption | Cloudflare Tunnel (TLS at edge) | In progress |
| Device token entropy | `secrets.token_urlsafe(32)` — 256-bit | Working |
| No hardcoded secrets | All secrets via env vars / auth store | Working |

## Gaps

### 1. Device tokens never expire

**Risk:** High
**Impact:** A compromised or leaked device token grants permanent access to the server. There is no expiration, rotation, or forced re-auth.

**Current behavior:** Once issued via `POST /api/auth/complete`, the device token is valid indefinitely unless the device is manually deleted from `auth_store.json`.

**Recommendation:**
- Add configurable token TTL (e.g. `AUTH_DEVICE_TOKEN_TTL_DAYS=90`)
- On expiry, require re-pairing or re-challenge
- Optionally: silent rotation — issue a new token on each successful connection, invalidate the old one

**Effort:** Medium — requires changes to `auth/store.py`, `auth/ws_auth.py`, and the mobile app (handle re-auth gracefully).

---

### 2. No remote device revocation

**Risk:** High
**Impact:** If a user loses a paired device (phone stolen, tablet lost), there is no way to revoke that device's access without manually editing `auth_store.json` or restarting the server.

**Current behavior:** The auth store has device records with a `status` field but no revocation API or UI.

**Recommendation:**
- Add `POST /api/auth/revoke` endpoint (authenticated, requires an active session from a different device or a master secret)
- Add device list endpoint `GET /api/auth/devices` so the user can see paired devices
- Consider a simple web UI or CLI command for device management
- The mobile app should show paired devices and allow remote revocation

**Effort:** Medium — the auth store already tracks devices; needs API endpoints + mobile app UI.

---

### 3. Setup code doesn't expire

**Risk:** Medium
**Impact:** The QR code / setup code generated at server startup is valid until the server restarts with `AUTH_REGENERATE_SETUP_CODE=true`. Anyone who photographs or receives the QR code can pair a new device at any time.

**Current behavior:** The bootstrap token in `auth_store.json` persists across restarts. `AUTH_REGENERATE_SETUP_CODE=true` forces a new one but must be manually set.

**Recommendation:**
- Add TTL to setup codes (e.g. 15 minutes after generation)
- One-time use: invalidate after first successful pairing
- Require explicit action to generate a new code (CLI command or button in a management UI)
- Show a countdown on the QR display so users know it will expire

**Effort:** Low — add `expires_at` to `BootstrapRecord` and check it in the pairing flow.

---

### 4. No brute-force protection on pairing

**Risk:** Medium
**Impact:** The setup code / bootstrap token is a base64-encoded JSON blob. While the token inside has high entropy (256-bit), the pairing endpoint itself has no lockout mechanism — an attacker could repeatedly attempt to pair.

**Current behavior:** Rate limiting exists on auth endpoints but there is no progressive lockout or permanent ban after repeated failures.

**Recommendation:**
- Lock the pairing endpoint after N failed attempts (e.g. 5 failures → 15 minute lockout)
- Log failed pairing attempts with source IP
- Alert the user (via the existing paired device) when a failed pairing attempt occurs

**Effort:** Low — extend the existing rate limiter with a failure counter.

---

### 5. No connection audit log

**Risk:** Low
**Impact:** No way to review who connected, when, from where, or how long sessions lasted. Makes incident investigation difficult.

**Current behavior:** Connection events are logged at INFO level to stdout/Docker logs but not persisted in a structured format.

**Recommendation:**
- Log connections to a structured audit file (`data/audit.jsonl`):
  ```json
  {"event": "connect", "device_id": "D7F0...", "ip": "1.2.3.4", "timestamp": "2026-03-22T10:00:00Z"}
  {"event": "disconnect", "device_id": "D7F0...", "duration_sec": 342, "timestamp": "2026-03-22T10:05:42Z"}
  {"event": "pair_attempt", "success": false, "ip": "5.6.7.8", "timestamp": "2026-03-22T11:00:00Z"}
  ```
- Expose via `GET /api/auth/audit` (authenticated)
- Retain last N days (configurable)

**Effort:** Low — wrap existing log calls with a file writer.

---

### 6. Tunnel token stored in plaintext

**Risk:** Low (acceptable for self-hosted)
**Impact:** If an attacker gains filesystem access to the server, they can read the Cloudflare Tunnel token from `.env` or `data/tunnel.json` and hijack the tunnel.

**Current behavior:** Tunnel token is an env var in `.env` or Docker Compose.

**Recommendation:**
- For most self-hosted users, this is acceptable (if they have filesystem access, they already own the machine)
- For higher-security deployments: use Docker secrets, systemd credentials, or a hardware security module
- Document the risk so users understand the trust boundary

**Effort:** N/A — document only, no code change needed.

---

### 7. No mutual TLS between tunnel and server

**Risk:** Low
**Impact:** The `cloudflared` → nyxclaw connection is plain HTTP over localhost/Docker network. If the Docker network is compromised, traffic could be intercepted.

**Current behavior:** Cloudflare terminates TLS at the edge, then `cloudflared` forwards to `http://server:8080` over the Docker bridge network.

**Recommendation:**
- For Docker deployments this is fine — the bridge network is isolated
- For deployments where cloudflared runs on a different host, add TLS between cloudflared and nyxclaw (uvicorn `--ssl-keyfile`)
- Not a priority for single-machine deployments

**Effort:** N/A for Docker. Low for remote cloudflared setups.

---

## Priority Matrix

| # | Gap | Risk | Effort | Priority |
|---|-----|------|--------|----------|
| 1 | Device tokens never expire | High | Medium | P1 |
| 2 | No remote device revocation | High | Medium | P1 |
| 3 | Setup code doesn't expire | Medium | Low | P2 |
| 4 | No brute-force protection on pairing | Medium | Low | P2 |
| 5 | No connection audit log | Low | Low | P3 |
| 6 | Tunnel token in plaintext | Low | N/A | Document only |
| 7 | No mutual TLS (localhost) | Low | N/A | Document only |

## App Store Requirements

For Google Play and Apple App Store distribution:

- **Transport encryption (TLS):** Required. Solved by Cloudflare Tunnel.
- **Auth model:** Not prescriptive — Apple/Google don't mandate a specific auth protocol, but they expect encrypted transport and reasonable security practices.
- **Privacy policy:** Required. Must disclose what data is collected (audio, transcripts).
- **Data at rest:** Not strictly required for app store approval but expected for enterprise/compliance.

The TLS gap (item 0, solved by Cloudflare Tunnel) was the only hard app store blocker. Gaps 1-5 are best-practice improvements that strengthen the product but won't prevent app store approval.
