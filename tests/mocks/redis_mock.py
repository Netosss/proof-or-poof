"""
MockRedis — synchronous in-memory Redis stand-in for unit tests.

Supports: get, set, setex, incr, expire, delete, sadd, sismember, scard, pipeline.
Expiry is enforced lazily on read.
"""

import time


class MockPipeline:
    def __init__(self, redis: "MockRedis"):
        self._redis = redis
        self._commands: list[tuple] = []

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

    def exec(self) -> list:
        results = []
        for cmd in self._commands:
            op = cmd[0]
            if op == "sismember":
                results.append(self._redis.sismember(cmd[1], cmd[2]))
            elif op == "scard":
                results.append(self._redis.scard(cmd[1]))
            elif op == "sadd":
                results.append(self._redis.sadd(cmd[1], cmd[2]))
            elif op == "expire":
                results.append(self._redis.expire(cmd[1], cmd[2]))
        return results


class MockRedis:
    def __init__(self):
        self._store: dict[str, object] = {}
        self._expiry: dict[str, float] = {}
        self._sets: dict[str, set] = {}

    def _expired(self, key: str) -> bool:
        if key in self._expiry and time.time() > self._expiry[key]:
            self._store.pop(key, None)
            self._expiry.pop(key, None)
            return True
        return False

    def get(self, key: str):
        if self._expired(key):
            return None
        return self._store.get(key)

    def set(self, key: str, value, ex: int | None = None, nx: bool = False) -> bool | None:
        if nx and key in self._store and not self._expired(key):
            return None
        self._store[key] = value
        if ex:
            self._expiry[key] = time.time() + ex
        return True

    def setex(self, key: str, seconds: int, value) -> bool:
        self._store[key] = value
        self._expiry[key] = time.time() + seconds
        return True

    def incr(self, key: str) -> int:
        val = int(self._store.get(key, 0)) + 1
        self._store[key] = str(val)
        return val

    def expire(self, key: str, seconds: int) -> int:
        if key in self._store:
            self._expiry[key] = time.time() + seconds
            return 1
        return 0

    def delete(self, key: str) -> int:
        existed = key in self._store
        self._store.pop(key, None)
        self._expiry.pop(key, None)
        return 1 if existed else 0

    def sismember(self, key: str, value) -> bool:
        return value in self._sets.get(key, set())

    def scard(self, key: str) -> int:
        return len(self._sets.get(key, set()))

    def sadd(self, key: str, value) -> int:
        if key not in self._sets:
            self._sets[key] = set()
        added = value not in self._sets[key]
        self._sets[key].add(value)
        return 1 if added else 0

    def pipeline(self) -> MockPipeline:
        return MockPipeline(self)
