"""Sliding-window rate limiter middleware."""

import time
from collections import defaultdict
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding-window rate limiter.

    Limits are per-client IP.  The window size and max requests are
    configurable via ``settings.rate_limit_window`` and
    ``settings.rate_limit_requests``.
    """

    def __init__(
        self, app: Any, max_requests: int | None = None, window_seconds: int | None = None
    ) -> None:
        super().__init__(app)
        self._max_requests = max_requests or settings.rate_limit_requests
        self._window = window_seconds or settings.rate_limit_window
        # client_ip -> list of request timestamps
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_rate_limited(self, client_ip: str) -> tuple[bool, int]:
        """Check if the client is rate-limited.

        Returns (is_limited, remaining_requests).
        """
        now = time.time()
        window_start = now - self._window

        # Prune old timestamps
        timestamps = self._requests[client_ip]
        self._requests[client_ip] = [t for t in timestamps if t > window_start]

        count = len(self._requests[client_ip])
        remaining = max(0, self._max_requests - count)

        if count >= self._max_requests:
            return True, 0

        self._requests[client_ip].append(now)
        return False, remaining - 1

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        is_limited, remaining = self._is_rate_limited(client_ip)

        if is_limited:
            logger.warning("rate_limit_exceeded", client_ip=client_ip, path=request.url.path)
            return Response(
                content='{"detail":"Rate limit exceeded. Try again later.","error_code":"RATE_LIMITED"}',
                status_code=429,
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit": str(self._max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Window": str(self._window),
                    "Retry-After": str(self._window),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(self._window)
        return response
