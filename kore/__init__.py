"""
Kore — Motor de Conocimiento Inteligente
═════════════════════════════════════════
Cache + comprensión + decisión en un solo paquete.
Cero dependencias externas obligatorias.

    from kore import Kore

    k = Kore("mi_dominio")
    k.ingest("datos.jsonl")
    r = k.ask("¿cómo se hace X?")

IAFiscal © 2026
"""

from kore.engine import Kore
from kore.models import (
    KoreResult,
    KoreEntry,
    MatchLevel,
    DomainConfig,
)
from kore.memory import SessionMemory, Session, Turn

__version__ = "0.2.0"
__all__ = [
    "Kore",
    "KoreResult",
    "KoreEntry",
    "MatchLevel",
    "DomainConfig",
    "SessionMemory",
    "Session",
    "Turn",
]
