"""
MockFirestore — synchronous in-memory Firestore stand-in for unit tests.

Supports: collection(), document(), get(), set(), update(), seed(), transaction().
SERVER_TIMESTAMP sentinels are silently dropped so tests can inspect real values.
"""

from unittest.mock import MagicMock


def _is_sentinel(value) -> bool:
    """Return True for Firestore sentinel objects (SERVER_TIMESTAMP, etc.)."""
    type_name = type(value).__name__
    return type_name in ("ServerTimestamp", "Sentinel", "_UNSET_SENTINEL")


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

    def get(self, transaction=None) -> MockDocumentSnapshot:
        data = self._store.get(self._doc_id)
        return MockDocumentSnapshot(data, exists=self._doc_id in self._store)

    def set(self, data: dict) -> None:
        clean = {k: v for k, v in data.items() if not _is_sentinel(v)}
        self._store[self._doc_id] = clean

    def update(self, data: dict) -> None:
        if self._doc_id not in self._store:
            self._store[self._doc_id] = {}
        for k, v in data.items():
            if not _is_sentinel(v):
                self._store[self._doc_id][k] = v


class MockCollection:
    def __init__(self):
        self._docs: dict[str, dict] = {}

    def document(self, doc_id: str) -> MockDocumentReference:
        return MockDocumentReference(self._docs, doc_id)


class MockFirestore:
    def __init__(self):
        self._collections: dict[str, MockCollection] = {}

    def collection(self, name: str) -> MockCollection:
        if name not in self._collections:
            self._collections[name] = MockCollection()
        return self._collections[name]

    def transaction(self) -> MagicMock:
        """Returns a MagicMock so transaction.set/update calls are silently absorbed."""
        return MagicMock()

    def seed(self, collection: str, doc_id: str, data: dict) -> None:
        """Pre-populate a document for test setup."""
        if collection not in self._collections:
            self._collections[collection] = MockCollection()
        self._collections[collection]._docs[doc_id] = dict(data)
