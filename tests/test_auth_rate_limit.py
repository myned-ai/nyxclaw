"""Tests for endpoint-keyed auth rate limiting behavior."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project src to path for direct test imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from auth.rate_limiter import RateLimiter


def test_endpoint_rate_limit_blocks_after_capacity() -> None:
    limiter = RateLimiter(endpoint_capacity=2, endpoint_refill_rate=0.0)

    allowed1, _ = limiter.check_endpoint_rate_limit("challenge", "1.2.3.4", "device-a")
    allowed2, _ = limiter.check_endpoint_rate_limit("challenge", "1.2.3.4", "device-a")
    allowed3, err3 = limiter.check_endpoint_rate_limit("challenge", "1.2.3.4", "device-a")

    assert allowed1 is True
    assert allowed2 is True
    assert allowed3 is False
    assert err3 is not None


def test_endpoint_rate_limit_isolated_by_endpoint_and_key() -> None:
    limiter = RateLimiter(endpoint_capacity=1, endpoint_refill_rate=0.0)

    a1, _ = limiter.check_endpoint_rate_limit("challenge", "1.2.3.4", "device-a")
    a2, _ = limiter.check_endpoint_rate_limit("challenge", "1.2.3.4", "device-b")
    a3, _ = limiter.check_endpoint_rate_limit("refresh", "session-a", "jti-a", "1.2.3.4")

    # Different device key and endpoint should not share same bucket.
    assert a1 is True
    assert a2 is True
    assert a3 is True
