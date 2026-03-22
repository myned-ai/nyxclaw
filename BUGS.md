# Known Bugs

## [ALL PLATFORMS] No TLS — app connects over plaintext WebSocket (ws://)

**Status:** Open
**Severity:** High
**Affects:** Android, iOS (future), Desktop (future)

### Problem

The app connects to the nyxclaw server over `ws://` (unencrypted). Android blocks cleartext traffic by default, requiring `cleartextTrafficPermitted="true"` as a workaround. This is not acceptable for marketplace distribution.

### Why it matters

- Unencrypted traffic can be intercepted on public/shared networks
- Google Play and Apple App Store expect secure connections
- Android's network security policy blocks cleartext by default (API 28+)
- iOS App Transport Security blocks cleartext by default

### Current workaround

Android: `network_security_config.xml` sets `cleartextTrafficPermitted="true"` globally.

### Solution needed

Decide on a secure connection strategy for a self-hosted server product:

1. **Auto-provisioned subdomain** (Plex model) — server registers as `<device-id>.nyxclaw.io`, auto-provisions a Let's Encrypt cert. Seamless for users, requires nyxclaw to run a DNS service.
2. **Tailscale** (OpenClaw model) — users install Tailscale on server + phone. Encrypted mesh VPN, stable DNS, zero nyxclaw infrastructure. Adds onboarding friction.
3. **TOFU + Tailscale hybrid** — pair on LAN via QR/mDNS, use Tailscale for remote access.

### DNS options for auto-provisioned subdomain (option 1)

| Service | Cost | Record limit | Notes |
|---------|------|-------------|-------|
| **Cloudflare** (free plan) | Domain only (~$10/yr) | 1,000 records | Full API, Let's Encrypt DNS-01 support, own domain (`*.d.nyxclaw.io`). Best fit — scales to paid tier when needed. |
| **deSEC** | Free (non-profit) | Unlimited | Full API, good wildcard cert support. Delegate a subdomain via NS records. |
| **Duck DNS** | Free | 5 subdomains/account | Simple HTTP API, Let's Encrypt support. No custom domain — subdomains under `duckdns.org`. |

**Recommended:** Cloudflare free plan. Each nyxclaw server calls the Cloudflare API on boot to register/update `<device-id>.d.nyxclaw.io` → its public IP, then provisions a Let's Encrypt cert via DNS-01 challenge.

### When resolved

- Android: revert `network_security_config.xml` base-config to `cleartextTrafficPermitted="false"`
- iOS: ensure ATS compliance (TLS required by default)
- Desktop: use `wss://` connections
