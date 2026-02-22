"""
Kore — Modelos de datos
═══════════════════════
Estructuras compartidas por todo el motor.

IAFiscal © 2026
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class MatchLevel(enum.Enum):
    """Nivel de matching que produjo el resultado."""
    EXACT = "exact"           # Hash exacto de query normalizada
    KEYWORDS = "keywords"     # Keywords + categoría
    BM25 = "bm25"             # Ranking BM25/TF-IDF
    SEMANTIC = "semantic"     # Mini-transformer ONNX (opcional)


@dataclass
class KoreEntry:
    """Una entrada de conocimiento en Kore."""
    query: str
    answer: str
    source: str = ""
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KoreResult:
    """Resultado de una consulta a Kore."""
    answer: str
    score: float
    level: MatchLevel
    source: str = ""
    category: str = ""
    entry_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def confident(self) -> bool:
        """¿Kore está seguro de esta respuesta?"""
        return self.score >= 0.7

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "score": self.score,
            "level": self.level.value,
            "source": self.source,
            "category": self.category,
            "confident": self.confident,
        }


@dataclass
class DomainConfig:
    """Configuración de un dominio de conocimiento."""
    name: str
    categories: dict[str, list[str]] = field(default_factory=dict)
    stopwords: list[str] = field(default_factory=list)
    min_score: float = 0.4
    learn_from_feedback: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> DomainConfig:
        return cls(
            name=data["name"],
            categories=data.get("categories", {}),
            stopwords=data.get("stopwords", []),
            min_score=data.get("min_score", 0.4),
            learn_from_feedback=data.get("learn_from_feedback", True),
        )


@dataclass
class KoreStats:
    """Estadísticas del motor."""
    total_entries: int = 0
    hits_exact: int = 0
    hits_keywords: int = 0
    hits_bm25: int = 0
    hits_semantic: int = 0
    misses: int = 0
    learned: int = 0

    @property
    def total_queries(self) -> int:
        return (self.hits_exact + self.hits_keywords +
                self.hits_bm25 + self.hits_semantic + self.misses)

    @property
    def hit_rate(self) -> float:
        total = self.total_queries
        if total == 0:
            return 0.0
        hits = (self.hits_exact + self.hits_keywords +
                self.hits_bm25 + self.hits_semantic)
        return hits / total

    def to_dict(self) -> dict:
        return {
            "total_entries": self.total_entries,
            "total_queries": self.total_queries,
            "hits": {
                "exact": self.hits_exact,
                "keywords": self.hits_keywords,
                "bm25": self.hits_bm25,
                "semantic": self.hits_semantic,
            },
            "misses": self.misses,
            "learned": self.learned,
            "hit_rate": f"{self.hit_rate:.1%}",
        }
