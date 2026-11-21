"""Shared pytest fixtures for deterministic local and CI test isolation."""

from __future__ import annotations

import contextlib
import os

import pytest
import redis.asyncio as aioredis

from core.config import get_settings

# Explicitly opt in to local dev persona tokens for the test suite (fail-closed in production)
os.environ.setdefault("ENABLE_DEV_TOKENS", "true")
get_settings.cache_clear()


@pytest.fixture(autouse=True)
async def _reset_redis_rate_limits() -> None:
    """Clear per-tenant Redis rate-limit counters before each test.

    The production API enforces a 30 runs/min per-tenant limit in Redis. Running the
    full 47-test suite back-to-back creates >30 runs for `tenant-alpha` within 60s,
    so resetting only `ratelimit:runs:*` keeps plain `pytest` deterministic without
    weakening the production rate limiter.
    """
    settings = get_settings()
    rds = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        with contextlib.suppress(Exception):
            keys = await rds.keys("ratelimit:runs:*")
            if keys:
                await rds.delete(*keys)
    finally:
        with contextlib.suppress(Exception):
            await rds.aclose()
