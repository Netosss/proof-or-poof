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

from unittest.mock import MagicMock


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
    def __init__(self, store: dict, doc_id: str):
        self._store = store
        self._doc_id = doc_id

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
        return MockSubCollection()


class MockSubCollection:
    """Minimal stub for subcollection (e.g. credit_ledger)."""

    def document(self, doc_id: str | None = None) -> MagicMock:
        """Returns a MagicMock so transaction.set(sub_ref, data) is absorbed."""
        return MagicMock()


class MockCollection:
    def __init__(self):
        self._docs: dict[str, dict] = {}

    def document(self, doc_id: str) -> MockDocumentReference:
        return MockDocumentReference(self._docs, doc_id)


class MockAsyncTransaction:
    """
    Minimal async transaction stub compatible with @async_transactional.

    @async_transactional checks `_id is None` and calls `_begin()`; then
    calls the wrapped function; then calls `_commit()`.  All three are
    fulfilled here without touching any real Firestore infrastructure.

    `set` / `update` / `delete` stage operations synchronously — they are
    absorbed silently so unit tests can focus on return-value assertions.
    """

    _id = None
    in_progress = True

    def set(self, ref, data, merge=False, **kwargs) -> None:
        pass

    def update(self, ref, data) -> None:
        pass

    def delete(self, ref) -> None:
        pass

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
