"""WebSocket challenge-response authentication.

Protocol:
1. Server sends:  {"event": "connect.challenge", "nonce": "<random>"}
2. Client sends:  {"rpc": "connect", "deviceId": "...", "publicKey": "...",
                   "signature": "...", "signedPayload": "...", "token": "..."}
3. Server verifies signature + registers device (TOFU) or validates device token
4. Server sends:  {"event": "connect.result", "deviceToken": "..."}
   or:            {"event": "connect.error", "code": "...", "message": "..."}
"""

from __future__ import annotations

import secrets
import time

from fastapi import WebSocket

from auth.device_verifier import DeviceVerificationError, DeviceVerifier
from auth.models import DeviceRecord, DeviceTokenRecord, utc_now
from auth.store import AuthStore, hash_token
from core.logger import get_logger

logger = get_logger(__name__)

CHALLENGE_TTL_SEC = 30
HANDSHAKE_TIMEOUT_SEC = 10
PROTOCOL_VERSION = "v3"


async def authenticate_websocket(
    websocket: WebSocket,
    auth_store: AuthStore,
    verifier: DeviceVerifier,
) -> dict[str, str] | None:
    """Run the challenge-response auth handshake on an already-accepted WebSocket.

    Returns an auth context dict on success, or None on failure (after sending error + closing).
    """
    nonce = secrets.token_urlsafe(24)
    nonce_id = secrets.token_urlsafe(12)
    now_ms = int(time.time() * 1000)
    expires_at_ms = now_ms + (CHALLENGE_TTL_SEC * 1000)

    auth_store.save_challenge_nonce(nonce_id, nonce, expires_at_ms)

    await websocket.send_json({
        "event": "connect.challenge",
        "nonce": nonce,
        "nonceId": nonce_id,
        "issuedAtMs": now_ms,
        "expiresIn": CHALLENGE_TTL_SEC,
        "algorithm": "ed25519",
    })

    try:
        msg = await _receive_with_timeout(websocket, HANDSHAKE_TIMEOUT_SEC)
    except Exception:
        await _send_error(websocket, "handshake_timeout", "Handshake timed out")
        return None

    if not isinstance(msg, dict) or msg.get("rpc") != "connect":
        await _send_error(websocket, "invalid_handshake", "Expected connect RPC")
        return None

    device_id = str(msg.get("deviceId", "")).strip()
    public_key = str(msg.get("publicKey", "")).strip()
    signature = str(msg.get("signature", "")).strip()
    signed_payload = str(msg.get("signedPayload", "")).strip()
    token = str(msg.get("token", "")).strip() or None
    algorithm = str(msg.get("algorithm", "ed25519")).strip().lower()

    if not device_id or not public_key or not signature or not signed_payload:
        await _send_error(websocket, "missing_fields", "Missing required auth fields")
        return None

    # Verify the nonce was used in the signed payload
    stored_nonce = auth_store.consume_challenge_nonce(nonce_id)
    if not stored_nonce:
        await _send_error(websocket, "expired_challenge", "Challenge expired or already used")
        return None

    # Verify signature
    try:
        verifier.verify_signature(
            signed_payload=signed_payload,
            signature_b64url=signature,
            public_key_b64url=public_key,
            algorithm=algorithm,
        )
    except DeviceVerificationError:
        await _send_error(websocket, "bad_signature", "Signature verification failed")
        return None

    # Check if nonce is present in the signed payload
    if stored_nonce not in signed_payload:
        await _send_error(websocket, "bad_signature", "Nonce not found in signed payload")
        return None

    # Look up existing device
    existing_device = auth_store.get_device(device_id)

    if existing_device:
        # Known device
        if existing_device.status == "revoked":
            await _send_error(websocket, "device_revoked", "Device has been revoked")
            return None

        if existing_device.public_key != public_key:
            await _send_error(websocket, "public_key_mismatch", "Public key does not match registered device")
            return None

        # Device has a token — verify it
        if token:
            if auth_store.verify_device_token(device_id, token):
                # Valid device token — reconnection
                await websocket.send_json({
                    "event": "connect.result",
                    "status": "ok",
                })
                return {"deviceId": device_id}

        # No token or invalid token — check bootstrap token for re-pairing
        if token and auth_store.verify_bootstrap_token(token):
            new_token = _issue_device_token(auth_store, device_id)
            await websocket.send_json({
                "event": "connect.result",
                "status": "ok",
                "deviceToken": new_token,
            })
            return {"deviceId": device_id}

        # Existing device with valid device token on file — try without presented token
        existing_dt = auth_store.get_device_token(device_id)
        if existing_dt and not existing_dt.revoked and not token:
            # Device is known but didn't present a token — require one
            await _send_error(websocket, "device_token_required", "Device token required for reconnection")
            return None

        await _send_error(websocket, "invalid_token", "Invalid or missing token")
        return None

    # New device — TOFU (trust on first use)
    if not token or not auth_store.verify_bootstrap_token(token):
        await _send_error(websocket, "bootstrap_required", "Valid bootstrap token required for new device registration")
        return None

    # Register new device
    auth_store.upsert_device(DeviceRecord(
        device_id=device_id,
        public_key=public_key,
        algorithm=algorithm,
        status="active",
        created_at=utc_now(),
    ))

    new_token = _issue_device_token(auth_store, device_id)
    await websocket.send_json({
        "event": "connect.result",
        "status": "ok",
        "deviceToken": new_token,
    })
    logger.info(f"New device registered via TOFU: {device_id[:12]}...")
    return {"deviceId": device_id}


def _issue_device_token(auth_store: AuthStore, device_id: str) -> str:
    """Generate and store a new device token, returning the raw token."""
    raw_token = secrets.token_urlsafe(32)
    auth_store.save_device_token(DeviceTokenRecord(
        device_id=device_id,
        token_hash=hash_token(raw_token),
        issued_at=utc_now(),
    ))
    return raw_token


async def _receive_with_timeout(websocket: WebSocket, timeout_sec: int) -> dict:
    import asyncio
    return await asyncio.wait_for(websocket.receive_json(), timeout=timeout_sec)


async def _send_error(websocket: WebSocket, code: str, message: str) -> None:
    logger.warning(f"Auth rejected: {code} — {message}")
    try:
        await websocket.send_json({
            "event": "connect.error",
            "code": code,
            "message": message,
        })
        await websocket.close(code=4401, reason=code)
    except Exception:
        pass
