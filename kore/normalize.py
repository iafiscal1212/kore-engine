"""
Kore — Normalización de texto
══════════════════════════════
Limpieza y normalización de queries para matching.
Soporte multiidioma (español, inglés, francés, alemán, portugués).
Puro Python, cero dependencias.

IAFiscal © 2026
"""

from __future__ import annotations

import re
import hashlib
import unicodedata


# ─────────────────────────────────────────────────────────────────────────────
# STOPWORDS multiidioma
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS_ES = frozenset([
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
    "al", "a", "en", "por", "para", "con", "sin", "que", "es", "son",
    "se", "su", "sus", "le", "les", "lo", "me", "te", "nos", "y", "o",
    "e", "u", "como", "cual", "este", "esta", "estos", "estas", "ese",
    "esa", "esos", "esas", "mi", "tu", "muy", "mas", "ya", "hay",
    "ser", "estar", "tener", "haber", "hacer", "poder", "deber",
    "no", "si", "pero", "cuando", "donde", "quien", "todo", "toda",
])

_STOPWORDS_EN = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "about", "between", "through", "during", "before", "after",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "what", "which", "who", "whom", "how", "when",
    "where", "why", "if", "then", "than", "very", "just", "also",
])

_STOPWORDS_ALL = _STOPWORDS_ES | _STOPWORDS_EN


def remove_accents(text: str) -> str:
    """Elimina acentos y diacríticos (ñ→n, ü→u, etc.)."""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')


def normalize(query: str, stopwords: frozenset | None = None) -> str:
    """
    Normaliza una query para matching:
      - minúsculas, sin acentos
      - sin puntuación (excepto % €)
      - sin stopwords
      - palabras ordenadas alfabéticamente
    """
    sw = stopwords if stopwords is not None else _STOPWORDS_ALL
    q = query.lower().strip()
    q = remove_accents(q)
    q = re.sub(r'[^\w\s%€]', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()
    words = [w for w in q.split() if w not in sw and len(w) > 1]
    return ' '.join(sorted(words))


def hash_text(text: str) -> str:
    """SHA256 truncado a 16 chars."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def tokenize(text: str, stopwords: frozenset | None = None) -> list[str]:
    """
    Tokeniza texto para BM25:
      - minúsculas, sin acentos, sin puntuación
      - sin stopwords
      - devuelve lista de tokens (NO ordenados)
    """
    sw = stopwords if stopwords is not None else _STOPWORDS_ALL
    t = text.lower().strip()
    t = remove_accents(t)
    t = re.sub(r'[^\w\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return [w for w in t.split() if w not in sw and len(w) > 1]


def extract_keywords(text: str, categories: dict[str, list[str]]) -> tuple[str, list[str]]:
    """
    Detecta categoría y extrae keywords de un texto según el dominio.

    Returns:
        (categoria, [keywords encontrados])
    """
    text_lower = remove_accents(text.lower())
    scores: dict[str, int] = {}
    found: list[str] = []

    for cat, keywords in categories.items():
        score = 0
        for kw in keywords:
            if kw in text_lower:
                score += 1
                if kw not in found:
                    found.append(kw)
        if score > 0:
            scores[cat] = score

    category = max(scores, key=scores.get) if scores else "general"
    return category, found[:15]
