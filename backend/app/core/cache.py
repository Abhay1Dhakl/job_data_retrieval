from __future__ import annotations

from functools import lru_cache
from typing import Optional

import redis
from redis import Redis

from app.core.config import Settings


@lru_cache
def _get_client(redis_url: str) -> Redis:
    """Create and memoize a Redis client for the given URL.

    Args:
        redis_url: Redis connection URL.
    Returns:
        A Redis client instance.
    """
    return redis.Redis.from_url(redis_url, decode_responses=True)


def get_cache(settings: Settings) -> Optional[Redis]:
    """Return a usable Redis client when configured and reachable.

    Args:
        settings: Application settings containing Redis configuration.
    Returns:
        A Redis client if available; otherwise None.
    """
    if not settings.redis_url:
        return None
    try:
        client = _get_client(settings.redis_url)
        client.ping()
        return client
    except Exception:
        return None
