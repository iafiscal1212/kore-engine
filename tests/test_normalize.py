"""Tests — Normalización de texto."""
from kore.normalize import normalize, hash_text, tokenize, extract_keywords, remove_accents


def test_remove_accents():
    assert remove_accents("café") == "cafe"
    assert remove_accents("niño") == "nino"
    assert remove_accents("über") == "uber"


def test_normalize_basic():
    result = normalize("¿Cómo se calcula el IVA?")
    assert "iva" in result
    assert "calcula" in result
    # Stopwords eliminadas
    assert "como" not in result
    assert "se" not in result
    assert "el" not in result


def test_normalize_sorted():
    r1 = normalize("IVA soportado compra")
    r2 = normalize("compra IVA soportado")
    assert r1 == r2


def test_normalize_accents_ignored():
    r1 = normalize("retención IRPF nómina")
    r2 = normalize("retencion irpf nomina")
    assert r1 == r2


def test_hash_deterministic():
    h1 = hash_text("test query")
    h2 = hash_text("test query")
    assert h1 == h2
    assert len(h1) == 16


def test_hash_different():
    h1 = hash_text("query one")
    h2 = hash_text("query two")
    assert h1 != h2


def test_tokenize():
    tokens = tokenize("El IVA soportado en una compra")
    assert "iva" in tokens
    assert "soportado" in tokens
    assert "compra" in tokens
    # Stopwords no incluidas
    assert "el" not in tokens
    assert "en" not in tokens
    assert "una" not in tokens


def test_extract_keywords():
    categories = {
        "iva": ["iva", "soportado", "repercutido", "factura"],
        "nomina": ["nomina", "salario", "seguridad social"],
    }
    cat, kws = extract_keywords("IVA soportado en factura", categories)
    assert cat == "iva"
    assert "iva" in kws
    assert "soportado" in kws


def test_extract_keywords_no_match():
    categories = {"iva": ["iva"], "nomina": ["nomina"]}
    cat, kws = extract_keywords("receta de tortilla", categories)
    assert cat == "general"
    assert kws == []
