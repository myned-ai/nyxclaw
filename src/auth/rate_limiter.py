"""
Rate Limiter

Implements token bucket algorithm for rate limiting:
- Per-domain limits (all users from same domain)
- Per-session limits (individual WebSocket connections)

Prevents abuse and ensures fair usage across all clients.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""

    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_refill: float = field(init=False)

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.time()

    def refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens

        Returns:
            True if tokens were available and consumed, False otherwise
        """
        self.refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    def __init__(
        self,
        domain_capacity: int = 100,
        domain_refill_rate: float = 10.0,
        session_capacity: int = 30,
        session_refill_rate: float = 5.0,
        endpoint_capacity: int = 40,
        endpoint_refill_rate: float = 8.0,
    ):
        """
        Initialize Rate Limiter

        Args:
            domain_capacity: Max tokens per domain bucket
            domain_refill_rate: Tokens refilled per second per domain
            session_capacity: Max tokens per session bucket
            session_refill_rate: Tokens refilled per second per session
        """
        self.domain_capacity = domain_capacity
        self.domain_refill_rate = domain_refill_rate
        self.session_capacity = session_capacity
        self.session_refill_rate = session_refill_rate
        self.endpoint_capacity = endpoint_capacity
        self.endpoint_refill_rate = endpoint_refill_rate

        # Domain buckets: origin -> TokenBucket
        self.domain_buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(self.domain_capacity, self.domain_refill_rate)
        )

        # Session buckets: session_id -> TokenBucket
        self.session_buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(self.session_capacity, self.session_refill_rate)
        )

        # Endpoint buckets: endpoint+key tuple hash -> TokenBucket
        self.endpoint_buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(self.endpoint_capacity, self.endpoint_refill_rate)
        )

    def check_rate_limit(self, origin: str, session_id: str) -> tuple[bool, str | None]:
        """
        Check if request should be rate limited

        Args:
            origin: Origin domain
            session_id: Unique session identifier

        Returns:
            Tuple of (is_allowed, error_message)
        """
        # Check domain-level limit
        domain_bucket = self.domain_buckets[origin]
        if not domain_bucket.consume():
            return False, f"Rate limit exceeded for domain: {origin}"

        # Check session-level limit
        session_bucket = self.session_buckets[session_id]
        if not session_bucket.consume():
            # Refund domain token since session failed
            domain_bucket.tokens += 1
            return False, f"Rate limit exceeded for session: {session_id}"

        return True, None

    def cleanup_old_sessions(self, active_session_ids: set):
        """
        Remove buckets for sessions that are no longer active

        Args:
            active_session_ids: Set of currently active session IDs
        """
        # Get sessions to remove (not in active set)
        sessions_to_remove = set(self.session_buckets.keys()) - active_session_ids

        for session_id in sessions_to_remove:
            del self.session_buckets[session_id]

    def get_stats(self) -> dict:
        """Get rate limiter statistics"""
        return {
            "domain_buckets": len(self.domain_buckets),
            "session_buckets": len(self.session_buckets),
            "endpoint_buckets": len(self.endpoint_buckets),
            "domain_capacity": self.domain_capacity,
            "session_capacity": self.session_capacity,
            "endpoint_capacity": self.endpoint_capacity,
        }

    def check_endpoint_rate_limit(self, endpoint: str, *key_parts: str) -> tuple[bool, str | None]:
        """Rate-limit an endpoint by a composite key.

        Example keys:
        - challenge: ip + device fingerprint
        - complete: ip + device fingerprint + challenge_id
        - refresh: session_id + refresh fingerprint + ip
        - websocket: ip + subject/session
        """
        normalized = [endpoint.strip().lower()]
        for part in key_parts:
            candidate = (part or "unknown").strip().lower()
            normalized.append(candidate if candidate else "unknown")

        bucket_key = "|".join(normalized)
        bucket = self.endpoint_buckets[bucket_key]
        if not bucket.consume():
            return False, f"Rate limit exceeded for endpoint: {endpoint}"
        return True, None
