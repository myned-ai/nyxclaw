"""Device signature verification for challenge payloads."""

from __future__ import annotations

import base64
from functools import lru_cache

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_der_public_key


class DeviceVerificationError(Exception):
    pass


def _b64url_decode(value: str) -> bytes:
    padding = "=" * ((4 - (len(value) % 4)) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("utf-8"))


class DeviceVerifier:
    def verify_signature(self, signed_payload: str, signature_b64url: str, public_key_b64url: str, algorithm: str) -> bool:
        algo = algorithm.strip().lower()
        if algo != "ed25519":
            raise DeviceVerificationError("unsupported_algorithm")

        try:
            public_key_der = _b64url_decode(public_key_b64url.strip())
            loaded_key = load_der_public_key(public_key_der)
            if not isinstance(loaded_key, Ed25519PublicKey):
                raise DeviceVerificationError("invalid_public_key")

            signature = _b64url_decode(signature_b64url.strip())
            loaded_key.verify(signature, signed_payload.encode("utf-8"))
            return True
        except DeviceVerificationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DeviceVerificationError("signature_verification_failed") from exc


@lru_cache(maxsize=1)
def get_device_verifier() -> DeviceVerifier:
    return DeviceVerifier()
