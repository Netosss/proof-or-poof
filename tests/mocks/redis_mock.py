"""
MockRedis — async in-memory Redis stand-in for unit tests.

All public methods that production code awaits are `async def` so that
`await rc.get(key)` etc. work correctly inside the async ASGI handlers
exercised by TestClient.

For synchronous test-setup code (seeding data before a request), use the
`seed(key, value)` helper or write directly to `._store`.

Supports: get, set, setex, incr, expire, ttl, delete, sadd, sismember,
          scard, publish, aclose, pipeline.
Expiry is enforced lazily on read.
"""

import time


class MockPipeline:
    def __init__(self, redis: "MockRedis"):
        self._redis = redis
        self._commands: list[tuple] = []

    def incr(self, key: str) -> "MockPipeline":
        self._commands.append(("incr", key))
        return self

    def sismember(self, key: str, value) -> "MockPipeline":
        self._commands.append(("sismember", key, value))
        return self

    def scard(self, key: str) -> "MockPipeline":
        self._commands.append(("scard", key))
        return self

    def sadd(self, key: str, value) -> "MockPipeline":
        self._commands.append(("sadd", key, value))
        return self

    def expire(self, key: str, seconds: int) -> "MockPipeline":
        self._commands.append(("expire", key, seconds))
        return self

    async def execute(self) -> list:
        """Execute queued commands and return results (async)."""
        results = []
        for cmd in self._commands:
            op = cmd[0]
            if op == "incr":
                results.append(self._redis._incr(cmd[1]))
            elif op == "sismember":
                results.append(self._redis._sismember(cmd[1], cmd[2]))
            elif op == "scard":
                results.append(self._redis._scard(cmd[1]))
            elif op == "sadd":
                results.append(self._redis._sadd(cmd[1], cmd[2]))
            elif op == "expire":
                results.append(self._redis._expire(cmd[1], cmd[2]))
        return results


class MockRedis:
    def __init__(self):
        self._store: dict[str, object] = {}
        self._expiry: dict[str, float] = {}
        self._sets: dict[str, set] = {}

    # ------------------------------------------------------------------
    # Internal sync helpers (used by MockPipeline and seed())
    # ------------------------------------------------------------------

    def _expired(self, key: str) -> bool:
        if key in self._expiry and time.time() > self._expiry[key]:
            self._store.pop(key, None)
            self._expiry.pop(key, None)
            return True
        return False

    def _incr(self, key: str) -> int:
        val = int(self._store.get(key, 0)) + 1
        self._store[key] = str(val)
        return val

    def _expire(self, key: str, seconds: int) -> int:
        if key in self._store or key in self._sets:
            self._expiry[key] = time.time() + seconds
            return 1
        return 0

    def _sismember(self, key: str, value) -> bool:
        return value in self._sets.get(key, set())

    def _scard(self, key: str) -> int:
        return len(self._sets.get(key, set()))

    def _sadd(self, key: str, value) -> int:
        if key not in self._sets:
            self._sets[key] = set()
        added = value not in self._sets[key]
        self._sets[key].add(value)
        return 1 if added else 0

    # ------------------------------------------------------------------
    # Sync helper for test setup (direct store access without await)
    # ------------------------------------------------------------------

    def seed(self, key: str, value, ex: int | None = None) -> None:
        """Synchronously seed a key for test setup (no await needed)."""
        self._store[key] = value
        if ex:
            self._expiry[key] = time.time() + ex

    # ------------------------------------------------------------------
    # Async public API (matches redis.asyncio interface)
    # ------------------------------------------------------------------

    async def get(self, key: str):
        if self._expired(key):
            return None
        return self._store.get(key)

    async def set(self, key: str, value, ex: int | None = None, nx: bool = False) -> bool | None:
        if nx and key in self._store and not self._expired(key):
            return None
        self._store[key] = value
        if ex:
            self._expiry[key] = time.time() + ex
        return True

    async def setex(self, key: str, seconds: int, value) -> bool:
        self._store[key] = value
        self._expiry[key] = time.time() + seconds
        return True

    async def incr(self, key: str) -> int:
        return self._incr(key)

    async def expire(self, key: str, seconds: int) -> int:
        return self._expire(key, seconds)

    async def ttl(self, key: str) -> int:
        if key not in self._expiry:
            return -1
        remaining = self._expiry[key] - time.time()
        return max(0, int(remaining))

    async def getdel(self, key: str):
        if self._expired(key):
            return None
        value = self._store.pop(key, None)
        self._expiry.pop(key, None)
        return value

    async def delete(self, key: str) -> int:
        existed = key in self._store
        self._store.pop(key, None)
        self._expiry.pop(key, None)
        return 1 if existed else 0

    async def sismember(self, key: str, value) -> bool:
        return self._sismember(key, value)

    async def scard(self, key: str) -> int:
        return self._scard(key)

    async def sadd(self, key: str, value) -> int:
        return self._sadd(key, value)

    async def publish(self, channel: str, message: str) -> int:
        """No real subscribers in tests — always returns 0."""
        return 0

    async def aclose(self) -> None:
        """No-op: nothing to close for an in-memory mock."""
        pass

    def pipeline(self) -> MockPipeline:
        return MockPipeline(self)
