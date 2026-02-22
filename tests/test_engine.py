"""Tests — Motor principal Kore."""
import json
import tempfile
from pathlib import Path

import pytest
from kore import Kore, KoreResult, MatchLevel, DomainConfig


@pytest.fixture
def engine():
    """Motor Kore en memoria para tests."""
    k = Kore(domain="test", db_path=":memory:")
    yield k
    k.close()


@pytest.fixture
def fiscal_engine():
    """Motor Kore con dominio fiscal."""
    config = DomainConfig(
        name="fiscal",
        categories={
            "iva": ["iva", "soportado", "repercutido", "factura", "tipo impositivo"],
            "nomina": ["nomina", "salario", "seguridad social", "cotizacion"],
            "contabilidad": ["asiento", "cuenta", "debe", "haber", "balance", "amortizacion"],
        },
        min_score=0.4,
    )
    k = Kore(domain="fiscal", db_path=":memory:", config=config)
    yield k
    k.close()


# ── Tests básicos ─────────────────────────────────────────────────────────

def test_create():
    k = Kore(domain="test", db_path=":memory:")
    assert repr(k).startswith("Kore(")
    k.close()


def test_empty_ask(engine):
    result = engine.ask("cualquier cosa")
    assert result is None


def test_add_and_ask(engine):
    engine.add("¿Qué es Python?", "Un lenguaje de programación")
    result = engine.ask("¿Qué es Python?")
    assert result is not None
    assert result.level == MatchLevel.EXACT
    assert result.score == 1.0
    assert "programación" in result.answer


def test_add_and_ask_normalized(engine):
    engine.add("¿Cómo se calcula el IVA?", "Se aplica el 21%")
    # Misma query con variaciones
    result = engine.ask("como se calcula el iva")
    assert result is not None
    assert result.score == 1.0


def test_miss(engine):
    engine.add("IVA soportado", "Respuesta sobre IVA")
    result = engine.ask("receta de tortilla")
    # Puede dar None o score bajo
    if result:
        assert result.score < 0.4 or result.level == MatchLevel.BM25


# ── Tests con keywords ────────────────────────────────────────────────────

def test_keyword_hit(fiscal_engine):
    fiscal_engine.add("Tipos de IVA en España", "21%, 10%, 4%", category="iva")
    fiscal_engine.add("Asiento de nómina", "Debe 640 Haber 476", category="nomina")

    result = fiscal_engine.ask("cuanto es el IVA soportado")
    assert result is not None
    assert result.level in (MatchLevel.EXACT, MatchLevel.KEYWORDS, MatchLevel.BM25)


def test_keyword_miss_non_fiscal(fiscal_engine):
    fiscal_engine.add("Tipos de IVA", "21%", category="iva")
    result = fiscal_engine.ask("receta de tortilla de patatas")
    assert result is None


# ── Tests BM25 ────────────────────────────────────────────────────────────

def test_bm25_similar_query(engine):
    engine.add("¿Cómo generar un asiento de compra con IVA?",
               "Debe: 600 Compras + 472 IVA soportado, Haber: 400 Proveedores")
    # Query similar pero no idéntica
    result = engine.ask("asiento compra IVA soportado")
    assert result is not None


# ── Tests ingesta ─────────────────────────────────────────────────────────

def test_ingest_jsonl():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(50):
            json.dump({"query": f"Pregunta numero {i:03d} sobre tema", "answer": f"Respuesta {i}"}, f)
            f.write('\n')
        path = f.name

    k = Kore(domain="test_ingest", db_path=":memory:")
    result = k.ingest(path)
    assert result['loaded'] == 50
    assert result['errors'] == 0
    assert k.stats().total_entries == 50
    k.close()
    Path(path).unlink()


def test_ingest_list(engine):
    data = [
        {"query": "Q1", "answer": "A1"},
        {"query": "Q2", "answer": "A2"},
        {"query": "Q3", "answer": "A3"},
    ]
    result = engine.ingest(data)
    assert result['loaded'] == 3


def test_ingest_csv():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("query,answer,source\n")
        f.write("Pregunta 1,Respuesta 1,test\n")
        f.write("Pregunta 2,Respuesta 2,test\n")
        path = f.name

    k = Kore(domain="test_csv", db_path=":memory:")
    result = k.ingest(path)
    assert result['loaded'] == 2
    k.close()
    Path(path).unlink()


def test_ingest_conversations():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        entry = {
            "conversations": [
                {"role": "user", "content": "¿Qué es el IVA?"},
                {"role": "assistant", "content": "Impuesto al Valor Añadido"},
            ]
        }
        json.dump(entry, f)
        f.write('\n')
        path = f.name

    k = Kore(domain="test_conv", db_path=":memory:")
    result = k.ingest(path)
    assert result['loaded'] == 1

    r = k.ask("¿Qué es el IVA?")
    assert r is not None
    assert "Valor Añadido" in r.answer
    k.close()
    Path(path).unlink()


# ── Tests feedback ────────────────────────────────────────────────────────

def test_feedback_learn(engine):
    engine.feedback("¿Qué es X?", "X es esto", positive=True)
    result = engine.ask("¿Qué es X?")
    assert result is not None
    assert result.source == "learned"


def test_feedback_negative(engine):
    initial_count = engine.stats().total_entries
    engine.feedback("mala query", "mala respuesta", positive=False)
    # No debería añadirse al conocimiento
    assert engine.stats().total_entries == initial_count


# ── Tests stats ───────────────────────────────────────────────────────────

def test_stats(engine):
    engine.add("test query", "test answer")
    engine.ask("test query")  # hit
    engine.ask("unknown")     # miss

    s = engine.stats()
    assert s.total_queries == 2
    assert s.hits_exact >= 1
    assert s.misses >= 1

    d = s.to_dict()
    assert "hit_rate" in d
    assert "total_entries" in d


# ── Tests context manager ────────────────────────────────────────────────

def test_context_manager():
    with Kore(domain="ctx", db_path=":memory:") as k:
        k.add("test", "test answer")
        assert k.ask("test") is not None


# ── Tests clear ───────────────────────────────────────────────────────────

def test_clear(engine):
    engine.add("q", "a")
    assert engine.stats().total_entries == 1
    engine.clear()
    assert engine.stats().total_entries == 0
