"""
Authentication Module

WebSocket challenge-response authentication with Ed25519 device keys.
"""

from .dependencies import get_auth_middleware
from .middleware import AuthMiddleware
from .rate_limiter import RateLimiter

__all__ = ["AuthMiddleware", "RateLimiter", "get_auth_middleware"]
