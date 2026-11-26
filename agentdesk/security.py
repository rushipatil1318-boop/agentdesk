"""Authentication (JWT) and rate limiting.

DEMO-GRADE identity: customers log in with their customer id and one shared demo password, the reviewer with
one reviewer password (both from environment variables). A real deployment would use an identity provider
(OIDC) instead; the point here is the *pattern*: the customer id comes from the signed token, never from the
request body, so a customer cannot impersonate another one.
"""
from __future__ import annotations

import hmac
import os
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import jwt

DEFAULT_CUSTOMER_PASSWORD = "demo-pass"
DEFAULT_REVIEWER_PASSWORD = "demo-reviewer-pass"


class AuthError(Exception):
    pass


@dataclass(frozen=True)
class Claims:
    subject: str
    role: str  # "customer" | "reviewer"


class TokenService:
    def __init__(self, customer_ids: set[str], secret: str | None = None, ttl_seconds: int = 3600,
                 customer_password: str = DEFAULT_CUSTOMER_PASSWORD,
                 reviewer_password: str = DEFAULT_REVIEWER_PASSWORD, clock=time.time) -> None:
        self.customer_ids = set(customer_ids)
        if secret is not None and len(secret) < 32:
            raise ValueError("JWT secret must be at least 32 characters long")
        self.secret = secret or secrets.token_urlsafe(32)  # per-process if not provided
        self.ttl = ttl_seconds
        self.customer_password = customer_password
        self.reviewer_password = reviewer_password
        self.clock = clock

    @classmethod
    def from_env(cls, customer_ids: set[str]) -> "TokenService":
        return cls(customer_ids, secret=os.getenv("JWT_SECRET") or None,
                   ttl_seconds=int(os.getenv("JWT_TTL_SECONDS", "3600")),
                   customer_password=os.getenv("DEMO_CUSTOMER_PASSWORD", DEFAULT_CUSTOMER_PASSWORD),
                   reviewer_password=os.getenv("REVIEWER_PASSWORD", DEFAULT_REVIEWER_PASSWORD))

    @property
    def default_credentials_in_use(self) -> bool:
        return (self.customer_password == DEFAULT_CUSTOMER_PASSWORD
                or self.reviewer_password == DEFAULT_REVIEWER_PASSWORD)

    def login(self, username: str, password: str) -> tuple[str, str]:
        if username == "reviewer":
            ok, role = hmac.compare_digest(password.encode(), self.reviewer_password.encode()), "reviewer"
        elif username in self.customer_ids:
            ok, role = hmac.compare_digest(password.encode(), self.customer_password.encode()), "customer"
        else:
            hmac.compare_digest(password.encode(), b"x")  # keep timing similar for unknown users
            ok, role = False, ""
        if not ok:
            raise AuthError("Invalid username or password")
        return self._issue(username, role), role

    def _issue(self, subject: str, role: str) -> str:
        now = int(self.clock())
        return jwt.encode({"sub": subject, "role": role, "iat": now, "exp": now + self.ttl},
                          self.secret, algorithm="HS256")

    def verify(self, token: str) -> Claims:
        try:
            data = jwt.decode(token, self.secret, algorithms=["HS256"],
                              options={"verify_exp": True, "require": ["sub", "role", "exp"]})
        except jwt.PyJWTError as exc:
            raise AuthError("Invalid or expired token") from exc
        if data["role"] not in ("customer", "reviewer"):
            raise AuthError("Invalid token role")
        return Claims(str(data["sub"]), str(data["role"]))


class RateLimiter:
    """Sliding-window limiter, in memory (per process). Use Redis or a gateway when running many replicas."""

    def __init__(self, limit: int, window_seconds: float = 60.0, clock=time.monotonic) -> None:
        self.limit = limit
        self.window = window_seconds
        self.clock = clock
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True
