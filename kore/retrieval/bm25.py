"""
Kore — BM25 Retrieval (Nivel 3)
════════════════════════════════
Implementación BM25 (Okapi) en Python puro.
Sin numpy, sin scikit-learn, sin nada externo.

BM25 es el algoritmo estándar de ranking en búsqueda de texto
(usado por Elasticsearch, Lucene, etc.) pero aquí cabe en 100 líneas.

IAFiscal © 2026
"""

from __future__ import annotations

import math
from typing import Optional


class BM25Index:
    """
    Índice BM25 in-memory.

    Uso:
        idx = BM25Index()
        idx.add("id1", ["token", "token", ...])
        idx.add("id2", ["token", "token", ...])
        idx.build()
        results = idx.search(["query", "tokens"], top_k=5)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self._k1 = k1
        self._b = b
        # doc_id -> tokens
        self._docs: dict[str, list[str]] = {}
        # token -> set of doc_ids
        self._inverted: dict[str, set[str]] = {}
        # doc_id -> len(tokens)
        self._doc_lens: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._n_docs: int = 0
        self._built = False

    @property
    def size(self) -> int:
        return self._n_docs

    def add(self, doc_id: str, tokens: list[str]):
        """Añade un documento al índice."""
        self._docs[doc_id] = tokens
        self._doc_lens[doc_id] = len(tokens)
        for token in set(tokens):
            if token not in self._inverted:
                self._inverted[token] = set()
            self._inverted[token].add(doc_id)
        self._built = False

    def build(self):
        """Calcula estadísticas globales. Llamar después de añadir docs."""
        self._n_docs = len(self._docs)
        if self._n_docs > 0:
            self._avgdl = sum(self._doc_lens.values()) / self._n_docs
        else:
            self._avgdl = 1.0
        self._built = True

    def search(self, query_tokens: list[str], top_k: int = 5,
               min_score: float = 0.0) -> list[tuple[str, float]]:
        """
        Busca los documentos más relevantes para la query.

        Returns:
            Lista de (doc_id, score) ordenada por score desc.
        """
        if not self._built:
            self.build()

        if self._n_docs == 0:
            return []

        scores: dict[str, float] = {}

        for token in query_tokens:
            if token not in self._inverted:
                continue

            doc_ids = self._inverted[token]
            # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            df = len(doc_ids)
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id in doc_ids:
                # TF del token en este documento
                tf = self._docs[doc_id].count(token)
                dl = self._doc_lens[doc_id]

                # BM25 score para este token-documento
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
                token_score = idf * numerator / denominator

                scores[doc_id] = scores.get(doc_id, 0.0) + token_score

        if not scores:
            return []

        # Normalizar scores a [0, 1]
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            normalized = {k: v / max_score for k, v in scores.items()}
        else:
            normalized = scores

        # Filtrar por min_score y ordenar
        results = [
            (doc_id, score) for doc_id, score in normalized.items()
            if score >= min_score
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def remove(self, doc_id: str):
        """Elimina un documento del índice."""
        if doc_id in self._docs:
            tokens = self._docs[doc_id]
            for token in set(tokens):
                if token in self._inverted:
                    self._inverted[token].discard(doc_id)
                    if not self._inverted[token]:
                        del self._inverted[token]
            del self._docs[doc_id]
            del self._doc_lens[doc_id]
            self._built = False

    def clear(self):
        """Limpia el índice completo."""
        self._docs.clear()
        self._inverted.clear()
        self._doc_lens.clear()
        self._n_docs = 0
        self._avgdl = 0.0
        self._built = False
