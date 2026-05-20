"""
MockFirestore — async in-memory Firestore stand-in for unit tests.

MockDocumentReference methods (get, set, update) are `async def` so that
`await ref.get()` works inside the async service functions called by the
ASGI handlers under test.

MockAsyncTransaction satisfies the interface expected by @async_transactional:
  - _id starts as None so the decorator calls _begin()
  - _begin/_commit/_rollback are async no-ops
  - set/update stage operations (absorbed silently so tests focus on return values)

SERVER_TIMESTAMP sentinels are silently dropped so tests can inspect real values.
"""

from unittest.mock import MagicMock  # noqa: F401  (kept for legacy callers)


def _is_sentinel(value) -> bool:
    """Return True for Firestore sentinel objects (SERVER_TIMESTAMP, etc.)."""
    type_name = type(value).__name__
    return type_name in ("ServerTimestamp", "Sentinel", "_UNSET_SENTINEL", "Increment")


class MockDocumentSnapshot:
    def __init__(self, data: dict | None = None, exists: bool = True):
        self.exists = exists
        self._data = dict(data) if data else {}

    def to_dict(self) -> dict:
        return dict(self._data)

    def get(self, key):
        return self._data.get(key)


class MockDocumentReference:
    def __init__(self, store: dict, doc_id: str, subcollections: dict | None = None):
        self._store = store
        self._doc_id = doc_id
        # Mirror google-cloud-firestore's public `id` attribute on DocumentReference.
        self.id = doc_id
        # Subcollection state is shared per (doc_id) — the parent MockCollection
        # owns the dict and passes the slot in so MockSubCollection state
        # survives across `document(id)` calls.
        self._subcollections = subcollections if subcollections is not None else {}

    async def get(self, transaction=None) -> MockDocumentSnapshot:
        data = self._store.get(self._doc_id)
        return MockDocumentSnapshot(data, exists=self._doc_id in self._store)

    async def set(self, data: dict, merge: bool = False) -> None:
        if merge and self._doc_id in self._store:
            existing = dict(self._store[self._doc_id])
            existing.update({k: v for k, v in data.items() if not _is_sentinel(v)})
            self._store[self._doc_id] = existing
        else:
            self._store[self._doc_id] = {k: v for k, v in data.items() if not _is_sentinel(v)}

    async def update(self, data: dict) -> None:
        if self._doc_id not in self._store:
            self._store[self._doc_id] = {}
        for k, v in data.items():
            if not _is_sentinel(v):
                self._store[self._doc_id][k] = v

    def collection(self, name: str) -> "MockSubCollection":
        if name not in self._subcollections:
            self._subcollections[name] = MockSubCollection()
        return self._subcollections[name]

    async def create(self, data: dict) -> None:
        """Mirror Firestore's create() — fails if doc already exists.

        Matches the production exception text so webhook handlers' duplicate
        detection (which checks for "ALREADY_EXISTS") triggers correctly.
        """
        if self._doc_id in self._store:
            raise Exception("409 ALREADY_EXISTS: document already exists")
        await self.set(data)


class _MockQuery:
    """Tiny chainable query stub: where(...).limit(...).stream()."""

    def __init__(self, source: "MockCollection | MockSubCollection"):
        self._source = source
        self._filters: list[tuple[str, str, object]] = []
        self._limit = None

    def where(self, field, op, value) -> "_MockQuery":
        self._filters.append((field, op, value))
        return self

    def limit(self, n: int) -> "_MockQuery":
        self._limit = n
        return self

    async def stream(self):
        docs = list(self._source._docs.items())
        for field, op, value in self._filters:
            if op != "==":
                continue
            docs = [(k, v) for k, v in docs if v.get(field) == value]
        if self._limit is not None:
            docs = docs[: self._limit]
        for doc_id, data in docs:
            snap = MockDocumentSnapshot(data, exists=True)
            snap.id = doc_id  # type: ignore[attr-defined]
            yield snap


class MockSubCollection:
    """Subcollection with full document semantics (used for api_credentials, credit_ledger)."""

    def __init__(self) -> None:
        self._docs: dict[str, dict] = {}

    def document(self, doc_id: str | None = None) -> "MockDocumentReference":
        if doc_id is None:
            import uuid as _uuid
            doc_id = _uuid.uuid4().hex
        return MockDocumentReference(self._docs, doc_id)

    def where(self, field, op, value) -> _MockQuery:
        return _MockQuery(self).where(field, op, value)

    def limit(self, n: int) -> _MockQuery:
        return _MockQuery(self).limit(n)

    async def stream(self):
        async for snap in _MockQuery(self).stream():
            yield snap


class MockCollection:
    def __init__(self):
        self._docs: dict[str, dict] = {}
        # Per-doc subcollection state, keyed by doc_id so it survives across
        # repeated `document(same_id)` calls.
        self._subcoll_state: dict[str, dict] = {}

    def document(self, doc_id: str | None = None) -> MockDocumentReference:
        if doc_id is None:
            import uuid as _uuid
            doc_id = _uuid.uuid4().hex
        if doc_id not in self._subcoll_state:
            self._subcoll_state[doc_id] = {}
        return MockDocumentReference(self._docs, doc_id, self._subcoll_state[doc_id])

    def where(self, field, op, value) -> _MockQuery:
        return _MockQuery(self).where(field, op, value)

    def limit(self, n: int) -> _MockQuery:
        return _MockQuery(self).limit(n)

    async def stream(self):
        async for snap in _MockQuery(self).stream():
            yield snap


class MockAsyncTransaction:
    """
    Minimal async transaction stub compatible with @async_transactional.

    @async_transactional checks `_id is None` and calls `_begin()`; then
    calls the wrapped function; then calls `_commit()`.  All three are
    fulfilled here without touching any real Firestore infrastructure.

    set/update/delete now apply to the underlying mock store so tests that
    inspect post-transaction state (e.g., credit_ledger queries for refund
    idempotency) see consistent values.
    """

    _id = None
    _read_only = False
    _max_attempts = 5
    in_progress = True

    def set(self, ref, data, merge=False, **kwargs) -> None:
        if hasattr(ref, "_store") and hasattr(ref, "_doc_id"):
            ref._store[ref._doc_id] = {k: v for k, v in data.items() if not _is_sentinel(v)}

    def update(self, ref, data) -> None:
        if hasattr(ref, "_store") and hasattr(ref, "_doc_id"):
            if ref._doc_id not in ref._store:
                ref._store[ref._doc_id] = {}
            for k, v in data.items():
                if not _is_sentinel(v):
                    ref._store[ref._doc_id][k] = v

    def delete(self, ref) -> None:
        if hasattr(ref, "_store") and hasattr(ref, "_doc_id"):
            ref._store.pop(ref._doc_id, None)

    async def _begin(self, retry_id=None) -> None:
        self._id = b"mock-txn-id"

    async def _commit(self) -> list:
        return []

    async def _rollback(self) -> None:
        pass

    def _clean_up(self) -> None:
        self._id = None
        self.in_progress = False


class MockFirestore:
    def __init__(self):
        self._collections: dict[str, MockCollection] = {}

    def collection(self, name: str) -> MockCollection:
        if name not in self._collections:
            self._collections[name] = MockCollection()
        return self._collections[name]

    def transaction(self) -> MockAsyncTransaction:
        """Returns a stub transaction compatible with @async_transactional."""
        return MockAsyncTransaction()

    def seed(self, collection: str, doc_id: str, data: dict) -> None:
        """Pre-populate a document for test setup."""
        if collection not in self._collections:
            self._collections[collection] = MockCollection()
        self._collections[collection]._docs[doc_id] = dict(data)
