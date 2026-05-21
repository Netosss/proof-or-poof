"""
One-shot Redis cleanup for legacy forensic cache keys.

Use after bumping the forensic cache prefix (see app/detection/cache.py
CACHE_PREFIX) if you want to reclaim memory immediately instead of waiting
for the 24h TTL on the old "forensic:*" keys to expire.

Usage:
    REDIS_URL=redis://... python scripts/redis_clear_forensic_cache.py [--dry-run]

The script SCANs in batches of 500 to avoid blocking the Redis instance,
then UNLINKs matched keys (non-blocking delete). Safe to run on production
during low traffic.
"""

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator

import redis.asyncio as redis_async

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("redis_clear_forensic_cache")

LEGACY_PATTERN = "forensic:*"
SCAN_BATCH = 500


async def _scan_keys(rc: "redis_async.Redis", pattern: str) -> AsyncIterator[bytes]:
    cursor = 0
    while True:
        cursor, batch = await rc.scan(cursor=cursor, match=pattern, count=SCAN_BATCH)
        for key in batch:
            yield key
        if cursor == 0:
            break


async def main(dry_run: bool) -> int:
    url = os.environ.get("REDIS_URL")
    if not url:
        logger.error("REDIS_URL env var is required")
        return 2

    rc = redis_async.from_url(url, decode_responses=False)

    total = 0
    deleted = 0
    pending: list[bytes] = []

    async for key in _scan_keys(rc, LEGACY_PATTERN):
        total += 1
        pending.append(key)

        if len(pending) >= SCAN_BATCH:
            if not dry_run:
                deleted += await rc.unlink(*pending)
            pending.clear()

    if pending and not dry_run:
        deleted += await rc.unlink(*pending)

    await rc.aclose()

    if dry_run:
        logger.info("dry_run_complete matched=%d (no deletes performed)", total)
    else:
        logger.info("cleanup_complete matched=%d unlinked=%d", total, deleted)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Count matches without deleting")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.dry_run)))
