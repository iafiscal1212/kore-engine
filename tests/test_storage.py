"""Tests — SQLite Storage."""
import pytest
from kore.storage.sqlite import SQLiteStore
from kore.models import KoreEntry


@pytest.fixture
def store():
    s = SQLiteStore(":memory:")
    yield s
    s.close()


def test_put_and_get(store):
    entry = KoreEntry(query="test query", answer="test answer", source="test")
    store.put("abc123", entry, "query test", ["test"], "general")
    result = store.get_by_id("abc123")
    assert result is not None
    assert result['answer'] == "test answer"


def test_get_missing(store):
    assert store.get_by_id("nonexistent") is None


def test_count(store):
    assert store.count() == 0
    entry = KoreEntry(query="q1", answer="a1")
    store.put("id1", entry, "q1", [], "general")
    assert store.count() == 1


def test_find_by_keywords(store):
    e1 = KoreEntry(query="IVA soportado", answer="resp1")
    e2 = KoreEntry(query="IVA repercutido", answer="resp2")
    store.put("id1", e1, "iva soportado", ["iva", "soportado"], "iva")
    store.put("id2", e2, "iva repercutido", ["iva", "repercutido"], "iva")

    results = store.find_by_keywords("iva", ["iva", "soportado"])
    assert len(results) >= 1
    # id1 debe tener más matches (2: iva + soportado)
    assert results[0][0] == "id1"
    assert results[0][1] == 2


def test_bulk_load(store):
    items = []
    for i in range(100):
        entry = KoreEntry(query=f"query {i}", answer=f"answer {i}")
        items.append((f"id_{i}", entry, f"query {i}", [], "general"))
    loaded, errors = store.put_bulk(items)
    assert loaded == 100
    assert errors == 0
    assert store.count() == 100


def test_stats(store):
    stats = store.get_stats()
    assert stats.total_entries == 0
    assert stats.total_queries == 0

    store.incr_stat("hits_exact")
    store.incr_stat("misses", 3)
    stats = store.get_stats()
    assert stats.hits_exact == 1
    assert stats.misses == 3


def test_feedback(store):
    store.save_feedback("test q", "test a", True, "test", "general")
    positive = store.get_positive_feedback(min_count=1)
    assert len(positive) == 1
    assert positive[0]['query'] == "test q"


def test_clear(store):
    entry = KoreEntry(query="q", answer="a")
    store.put("id1", entry, "q", ["kw"], "cat")
    store.incr_stat("hits_exact", 5)
    store.clear()
    assert store.count() == 0
    stats = store.get_stats()
    assert stats.hits_exact == 0
