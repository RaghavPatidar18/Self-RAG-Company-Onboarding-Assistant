"""Shared request pacing for Groq calls made by evaluation runs."""
from __future__ import annotations

from functools import lru_cache

from langchain_core.rate_limiters import InMemoryRateLimiter

from src.config.settings import settings


@lru_cache(maxsize=1)
def get_eval_rate_limiter() -> InMemoryRateLimiter:
    """Allow one eval model request per configured interval, with no burst."""
    interval = max(settings.eval_request_interval_seconds, 0.1)
    return InMemoryRateLimiter(
        requests_per_second=1.0 / interval,
        check_every_n_seconds=min(0.1, interval),
        max_bucket_size=1,
    )
