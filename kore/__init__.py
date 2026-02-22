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

__version__ = "0.1.0"
__all__ = [
    "Kore",
    "KoreResult",
    "KoreEntry",
    "MatchLevel",
    "DomainConfig",
]
