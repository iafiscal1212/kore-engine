"""
Kore — Memoria Conversacional
══════════════════════════════
Memoria de sesión por usuario/conversación.

Qué hace:
  - Recuerda las últimas N interacciones de cada sesión
  - Mantiene el contexto temático (de qué se está hablando)
  - Resuelve referencias implícitas ("y el repercutido?" → IVA repercutido)
  - Enriquece la query con contexto antes de buscar

Cómo funciona:
  1. El usuario pregunta "tipos de IVA"
  2. Kore responde y guarda en memoria: session=abc, topic=iva, entities=[IVA]
  3. El usuario dice "y el repercutido?"
  4. Kore ve que es vaga, mira la memoria → topic=iva, entities=[IVA]
  5. Enriquece: "y el repercutido?" → "IVA repercutido"
  6. Busca con la query enriquecida → hit

Puro Python, sin dependencias. Memoria en dict (volátil) o SQLite (persistente).

IAFiscal © 2026
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from typing import Optional

from kore.normalize import remove_accents


# ─────────────────────────────────────────────────────────────────────────────
# MODELOS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """Un turno de conversación."""
    query: str
    answer: str
    category: str = ""
    keywords: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    """Estado de una sesión conversacional."""
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    topic: str = ""               # Tema actual (categoría dominante)
    entities: list[str] = field(default_factory=list)  # Entidades mencionadas
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turns": self.turn_count,
            "topic": self.topic,
            "entities": self.entities,
            "last_active": self.last_active,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DETECCIÓN DE QUERIES VAGAS
# ─────────────────────────────────────────────────────────────────────────────

# Patrones que indican referencia al turno anterior
_REFERENCE_PATTERNS = [
    # Español
    r"^y\s+(el|la|los|las|lo)\b",       # "y el repercutido?"
    r"^y\s+(?:qué|que)\b",              # "y qué pasa con..."
    r"^(?:qué|que)\s+(?:más|mas)\b",    # "qué más?"
    r"^(?:también|tambien)\b",           # "también quiero..."
    r"^(?:además|ademas)\b",             # "además de eso..."
    r"^(?:ese|esa|eso|esto|este|esta)\b",  # "ese modelo..."
    r"^(?:el|la|lo)\s+(?:mismo|misma)\b",  # "lo mismo pero..."
    r"^(?:cuánto|cuanto|cuándo|cuando)\b(?!\s+(?:es|son|cuesta|vale))",  # "cuánto?" (sin verbo = referencia)
    r"^(?:por qué|porque|porqué)\s*\??\s*$",  # "por qué?" solo
    r"^(?:explica|detalla|amplía|amplia)\b",   # "explica más"
    r"^(?:cómo|como)\s+(?:se\s+)?(?:haría|haria|hace)\s*\??\s*$",  # "cómo se hace?" solo
    # Inglés
    r"^(?:and|what about)\s+(?:the|its?)\b",
    r"^(?:also|too)\b",
    r"^(?:that|this|those|these)\b",
    r"^(?:how|why|when)\s*\??\s*$",      # "how?" solo
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _REFERENCE_PATTERNS]


def is_vague_query(query: str) -> bool:
    """¿Es una query que necesita contexto para entenderse?"""
    q = query.strip()
    # Muy corta (1-3 palabras) → probablemente referencia
    if len(q.split()) <= 3 and '?' in q:
        return True
    # Matchea algún patrón de referencia
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(q):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class SessionMemory:
    """
    Gestor de memoria conversacional.

    Uso:
        mem = SessionMemory(max_turns=10)

        # Antes de buscar: enriquecer la query con contexto
        enriched = mem.enrich("y el repercutido?", session_id="user_123")

        # Después de responder: registrar el turno
        mem.record(session_id="user_123",
                   query="IVA repercutido", answer="...",
                   category="iva", keywords=["iva", "repercutido"])
    """

    def __init__(self, max_turns: int = 10, session_ttl: int = 3600):
        """
        Args:
            max_turns: Máximo de turnos por sesión (los más antiguos se descartan)
            session_ttl: Segundos de inactividad antes de expirar la sesión
        """
        self._sessions: dict[str, Session] = {}
        self._max_turns = max_turns
        self._session_ttl = session_ttl

    def enrich(self, query: str, session_id: str) -> str:
        """
        Enriquece una query con contexto de la sesión.

        Si la query es vaga ("y el repercutido?") y hay contexto previo,
        la expande con el tema y entidades de la conversación.

        Si la query es clara, la devuelve tal cual.
        """
        session = self._get_session(session_id)
        if not session or not session.turns:
            return query

        if not is_vague_query(query):
            return query

        # Construir contexto desde la sesión
        enriched = self._expand_query(query, session)
        return enriched

    def record(self, session_id: str, query: str, answer: str,
               category: str = "", keywords: list[str] | None = None):
        """Registra un turno en la sesión."""
        session = self._get_or_create_session(session_id)

        turn = Turn(
            query=query,
            answer=answer,
            category=category,
            keywords=keywords or [],
        )
        session.turns.append(turn)

        # Truncar si excede max_turns
        if len(session.turns) > self._max_turns:
            session.turns = session.turns[-self._max_turns:]

        # Actualizar topic y entities
        if category:
            session.topic = category
        if keywords:
            # Mantener últimas 20 entidades únicas
            for kw in keywords:
                if kw not in session.entities:
                    session.entities.append(kw)
            session.entities = session.entities[-20:]

        session.last_active = time.time()

    def get_context(self, session_id: str) -> dict:
        """Devuelve el contexto actual de una sesión."""
        session = self._get_session(session_id)
        if not session:
            return {"topic": "", "entities": [], "turns": 0}
        return session.to_dict()

    def get_history(self, session_id: str, last_n: int = 5) -> list[dict]:
        """Devuelve los últimos N turnos de una sesión."""
        session = self._get_session(session_id)
        if not session:
            return []
        turns = session.turns[-last_n:]
        return [
            {"query": t.query, "answer": t.answer[:200],
             "category": t.category}
            for t in turns
        ]

    def clear_session(self, session_id: str):
        """Elimina una sesión."""
        self._sessions.pop(session_id, None)

    def active_sessions(self) -> int:
        """Número de sesiones activas."""
        self._cleanup()
        return len(self._sessions)

    # ── INTERNOS ──────────────────────────────────────────────────────────

    def _get_session(self, session_id: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if session:
            # Comprobar TTL
            if time.time() - session.last_active > self._session_ttl:
                del self._sessions[session_id]
                return None
        return session

    def _get_or_create_session(self, session_id: str) -> Session:
        session = self._get_session(session_id)
        if not session:
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
        return session

    def _expand_query(self, query: str, session: Session) -> str:
        """Expande una query vaga con contexto de la sesión."""
        parts = []

        # Añadir topic como prefijo
        if session.topic:
            parts.append(session.topic)

        # Añadir últimas entidades relevantes (máx 5)
        recent_entities = session.entities[-5:]
        if recent_entities:
            parts.extend(recent_entities)

        # Añadir query original
        parts.append(query)

        # También incluir la query del último turno si es muy reciente
        if session.turns:
            last_turn = session.turns[-1]
            # Solo si fue hace menos de 5 minutos
            if time.time() - last_turn.timestamp < 300:
                parts.insert(0, last_turn.query)

        expanded = " ".join(parts)
        return expanded

    def _cleanup(self):
        """Elimina sesiones expiradas."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > self._session_ttl
        ]
        for sid in expired:
            del self._sessions[sid]
