from core.settings import get_settings

from .middleware import AuthMiddleware

_auth_middleware: AuthMiddleware | None = None


def get_auth_middleware() -> AuthMiddleware | None:
    global _auth_middleware

    if _auth_middleware:
        return _auth_middleware

    settings = get_settings()

    if not settings.auth_enabled:
        return None

    _auth_middleware = AuthMiddleware(
        enable_rate_limiting=settings.auth_enable_rate_limiting,
    )

    return _auth_middleware
