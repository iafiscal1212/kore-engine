"""Tests — BM25 Retrieval."""
from kore.retrieval.bm25 import BM25Index


def test_empty_index():
    idx = BM25Index()
    idx.build()
    assert idx.search(["test"]) == []


def test_add_and_search():
    idx = BM25Index()
    idx.add("doc1", ["iva", "soportado", "compra", "factura"])
    idx.add("doc2", ["nomina", "salario", "seguridad", "social"])
    idx.add("doc3", ["iva", "repercutido", "venta", "factura"])
    idx.build()

    results = idx.search(["iva", "soportado"], top_k=3)
    assert len(results) > 0
    # doc1 debe ser el mejor match (tiene iva + soportado)
    assert results[0][0] == "doc1"


def test_ranking_order():
    idx = BM25Index()
    idx.add("d1", ["python", "programacion", "web"])
    idx.add("d2", ["python", "programacion", "datos", "ciencia"])
    idx.add("d3", ["java", "programacion", "web"])
    idx.build()

    results = idx.search(["python", "datos", "ciencia"])
    # d2 tiene 3 matches, d1 tiene 1, d3 tiene 0
    assert results[0][0] == "d2"


def test_min_score():
    idx = BM25Index()
    idx.add("d1", ["iva", "soportado"])
    idx.add("d2", ["nomina", "salario"])
    idx.build()

    results = idx.search(["iva"], min_score=0.0)
    assert len(results) >= 1

    results_strict = idx.search(["iva"], min_score=0.99)
    # Con score normalizado, solo el top puede llegar a 1.0
    assert len(results_strict) <= 1


def test_remove():
    idx = BM25Index()
    idx.add("d1", ["test", "one"])
    idx.add("d2", ["test", "two"])
    idx.build()
    assert idx.size == 2

    idx.remove("d1")
    idx.build()
    assert idx.size == 1

    results = idx.search(["test"])
    assert len(results) == 1
    assert results[0][0] == "d2"


def test_clear():
    idx = BM25Index()
    idx.add("d1", ["test"])
    idx.build()
    idx.clear()
    assert idx.size == 0
    assert idx.search(["test"]) == []


def test_single_doc():
    idx = BM25Index()
    idx.add("d1", ["unica", "entrada", "prueba"])
    idx.build()
    results = idx.search(["unica", "prueba"])
    assert len(results) == 1
    assert results[0][0] == "d1"
    assert results[0][1] > 0  # Score positivo
