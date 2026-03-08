"""
Authentication Middleware

Security Implementation:
1. Rate Limiting - Token bucket algorithm
2. Monitoring - Audit logging
"""

import logging
import hashlib

from fastapi import WebSocket

from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class AuthMiddleware:
    def __init__(self, enable_rate_limiting: bool = True):
        """
        Initialize Authentication Middleware

        Args:
            enable_rate_limiting: Enable rate limiting
        """
        self.rate_limiter = RateLimiter() if enable_rate_limiting else None
        logger.info("AuthMiddleware initialized")

    def get_stats(self) -> dict:
        """Get authentication statistics"""
        stats = {"rate_limiter_enabled": self.rate_limiter is not None}

        if self.rate_limiter:
            stats["rate_limiter"] = self.rate_limiter.get_stats()

        return stats

    def check_endpoint_rate_limit(self, endpoint: str, *key_parts: str) -> tuple[bool, str | None]:
        """Apply endpoint-scoped rate limiting with redacted diagnostic logging."""
        if not self.rate_limiter:
            return True, None

        is_allowed, error = self.rate_limiter.check_endpoint_rate_limit(endpoint, *key_parts)
        if not is_allowed:
            redacted_key = self._redact_key_parts(*key_parts)
            logger.warning(f"Rate limit blocked endpoint={endpoint} key={redacted_key}")
        return is_allowed, error

    @staticmethod
    def _redact_key_parts(*parts: str) -> str:
        joined = "|".join((part or "").strip() for part in parts)
        digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
        return f"sha256:{digest}"
