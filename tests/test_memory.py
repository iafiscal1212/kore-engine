"""Tests — Memoria conversacional."""
import time
import pytest
from kore import Kore, DomainConfig
from kore.memory import SessionMemory, is_vague_query


# ── Tests is_vague_query ──────────────────────────────────────────────────

def test_vague_y_el():
    assert is_vague_query("y el repercutido?") is True

def test_vague_y_que():
    assert is_vague_query("y qué pasa con eso?") is True

def test_vague_tambien():
    assert is_vague_query("también quiero saber") is True

def test_vague_short_question():
    assert is_vague_query("cuánto?") is True

def test_vague_explica():
    assert is_vague_query("explica más") is True

def test_not_vague_full_question():
    assert is_vague_query("¿Cuáles son los tipos de IVA en España?") is False

def test_not_vague_long():
    assert is_vague_query("necesito hacer un asiento de compra con IVA soportado") is False

def test_vague_english():
    assert is_vague_query("and the other one?") is True

def test_vague_como_se_hace():
    assert is_vague_query("cómo se hace?") is True


# ── Tests SessionMemory ───────────────────────────────────────────────────

def test_memory_enrich_no_context():
    mem = SessionMemory()
    # Sin contexto previo, devuelve la query tal cual
    result = mem.enrich("y el repercutido?", "session1")
    assert result == "y el repercutido?"


def test_memory_record_and_enrich():
    mem = SessionMemory()
    # Simular conversación
    mem.record("s1", "tipos de IVA en España", "21%, 10%, 4%",
               category="iva", keywords=["iva", "tipo impositivo"])

    # Query vaga → debe enriquecerse
    enriched = mem.enrich("y el repercutido?", "s1")
    assert "iva" in enriched.lower()
    assert "repercutido" in enriched.lower()


def test_memory_clear_question():
    mem = SessionMemory()
    mem.record("s1", "IVA soportado", "respuesta...",
               category="iva", keywords=["iva", "soportado"])

    # Query clara → NO se enriquece
    enriched = mem.enrich("¿Cómo se calcula la nómina con seguridad social?", "s1")
    assert enriched == "¿Cómo se calcula la nómina con seguridad social?"


def test_memory_context():
    mem = SessionMemory()
    mem.record("s1", "IVA", "resp", category="iva", keywords=["iva"])
    ctx = mem.get_context("s1")
    assert ctx["topic"] == "iva"
    assert ctx["turns"] == 1


def test_memory_history():
    mem = SessionMemory()
    mem.record("s1", "Q1", "A1")
    mem.record("s1", "Q2", "A2")
    mem.record("s1", "Q3", "A3")

    history = mem.get_history("s1", last_n=2)
    assert len(history) == 2
    assert history[0]["query"] == "Q2"
    assert history[1]["query"] == "Q3"


def test_memory_max_turns():
    mem = SessionMemory(max_turns=3)
    for i in range(5):
        mem.record("s1", f"Q{i}", f"A{i}")

    history = mem.get_history("s1", last_n=10)
    assert len(history) == 3  # Solo los últimos 3


def test_memory_session_ttl():
    mem = SessionMemory(session_ttl=1)  # 1 segundo
    mem.record("s1", "Q", "A", category="iva", keywords=["iva"])

    time.sleep(1.1)
    ctx = mem.get_context("s1")
    assert ctx["turns"] == 0  # Sesión expirada


def test_memory_forget():
    mem = SessionMemory()
    mem.record("s1", "Q", "A")
    mem.clear_session("s1")
    assert mem.get_context("s1")["turns"] == 0


def test_memory_active_sessions():
    mem = SessionMemory()
    mem.record("s1", "Q1", "A1")
    mem.record("s2", "Q2", "A2")
    assert mem.active_sessions() == 2


def test_memory_entities_accumulate():
    mem = SessionMemory()
    mem.record("s1", "Q1", "A1", keywords=["iva", "soportado"])
    mem.record("s1", "Q2", "A2", keywords=["factura", "rectificativa"])

    ctx = mem.get_context("s1")
    assert "iva" in ctx["entities"]
    assert "factura" in ctx["entities"]


# ── Tests integración Kore + Memoria ─────────────────────────────────────

@pytest.fixture
def kore_fiscal():
    config = DomainConfig(
        name="fiscal",
        categories={
            "iva": ["iva", "soportado", "repercutido", "factura", "tipo impositivo"],
            "nomina": ["nomina", "salario", "seguridad social"],
            "contabilidad": ["asiento", "cuenta", "amortizacion", "leasing"],
        },
        min_score=0.4,
    )
    k = Kore(domain="test_mem", db_path=":memory:", config=config)
    # Cargar datos de prueba
    k.ingest([
        {"query": "Tipos de IVA en España", "answer": "General 21%, reducido 10%, superreducido 4%"},
        {"query": "IVA repercutido", "answer": "Es el IVA que cobra una empresa al vender. Cuenta 477."},
        {"query": "IVA soportado", "answer": "Es el IVA que paga una empresa al comprar. Cuenta 472."},
        {"query": "Asiento de nómina", "answer": "Debe: 640 Sueldos. Haber: 476 SS, 4751 IRPF, 465 Rem.ptes."},
        {"query": "Amortización de inmovilizado", "answer": "Debe: 681 Amort. inmov. material. Haber: 281."},
    ])
    yield k
    k.close()


def test_kore_ask_with_session(kore_fiscal):
    # Turno 1: query clara
    r1 = kore_fiscal.ask("tipos de IVA en España", session_id="user_1")
    assert r1 is not None
    assert "21%" in r1.answer

    # Turno 2: query vaga → Kore enriquece con contexto (topic=iva)
    r2 = kore_fiscal.ask("y el repercutido?", session_id="user_1")
    assert r2 is not None
    assert "477" in r2.answer or "repercutido" in r2.answer.lower()


def test_kore_ask_without_session(kore_fiscal):
    # Sin session_id, "y el repercutido?" no tiene contexto → miss o hit parcial
    r = kore_fiscal.ask("y el repercutido?")
    # Puede ser None o un hit parcial, pero no debería dar la respuesta de IVA repercutido
    # con score alto, porque no hay contexto


def test_kore_context(kore_fiscal):
    kore_fiscal.ask("tipos de IVA", session_id="s1")
    ctx = kore_fiscal.context("s1")
    assert ctx["topic"] == "iva"
    assert ctx["turns"] == 1


def test_kore_history(kore_fiscal):
    kore_fiscal.ask("tipos de IVA", session_id="s1")
    kore_fiscal.ask("asiento de nómina", session_id="s1")
    history = kore_fiscal.history("s1")
    assert len(history) == 2


def test_kore_forget(kore_fiscal):
    kore_fiscal.ask("tipos de IVA", session_id="s1")
    kore_fiscal.forget("s1")
    assert kore_fiscal.context("s1")["turns"] == 0


def test_kore_multi_session(kore_fiscal):
    """Dos sesiones independientes no se mezclan."""
    kore_fiscal.ask("tipos de IVA", session_id="user_a")
    kore_fiscal.ask("asiento de nómina", session_id="user_b")

    ctx_a = kore_fiscal.context("user_a")
    ctx_b = kore_fiscal.context("user_b")
    assert ctx_a["topic"] == "iva"
    assert ctx_b["topic"] == "nomina"
