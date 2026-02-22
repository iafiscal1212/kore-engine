"""
Kore — Motor Principal
══════════════════════
El cerebro que conecta todo: storage, retrieval, normalización, decisión.

    from kore import Kore

    k = Kore("mi_dominio")
    k.ingest("datos.jsonl")
    r = k.ask("¿cómo se hace X?")

IAFiscal © 2026
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from kore.models import (
    KoreEntry, KoreResult, KoreStats,
    MatchLevel, DomainConfig,
)
from kore.normalize import (
    normalize, hash_text, tokenize, extract_keywords,
)
from kore.storage.sqlite import SQLiteStore
from kore.retrieval.bm25 import BM25Index
from kore.ingest.loader import load_jsonl, load_csv, load_entries
from kore.memory import SessionMemory

logger = logging.getLogger(__name__)


class Kore:
    """
    Motor de Conocimiento Inteligente.

    4 niveles de retrieval:
      1. Hash exacto         (<0.1ms, score=1.0)
      2. Keywords+categoría  (<1ms)
      3. BM25 ranking        (<5ms)
      4. Semántico ONNX      (<15ms, opcional)

    Memoria conversacional:
      - Recuerda contexto por sesión (topic, entidades, historial)
      - Resuelve queries vagas ("y el repercutido?" → "IVA repercutido")
      - Configurable: max_turns, session_ttl
    """

    def __init__(
        self,
        domain: str = "default",
        db_path: str | Path | None = None,
        config: DomainConfig | None = None,
        semantic: bool = False,
        model_dir: str | Path | None = None,
        max_turns: int = 10,
        session_ttl: int = 3600,
    ):
        self.domain = domain
        self._config = config or DomainConfig(name=domain)

        # Storage
        if db_path is None:
            db_path = Path(f"kore_{domain}.db")
        self._store = SQLiteStore(db_path)

        # BM25 index (se construye al ingestar o al primer ask)
        self._bm25 = BM25Index()
        self._bm25_dirty = True

        # Semantic index (opcional)
        self._semantic = None
        if semantic:
            self._init_semantic(model_dir)

        # Memoria conversacional
        self._memory = SessionMemory(max_turns=max_turns, session_ttl=session_ttl)

        # Cargar BM25 desde entries existentes
        self._rebuild_bm25()

        entries = self._store.count()
        if entries > 0:
            logger.info(f"Kore [{domain}]: {entries} entries cargadas")

    def _init_semantic(self, model_dir=None):
        """Intenta inicializar el nivel semántico."""
        try:
            from kore.retrieval.semantic import SemanticIndex, is_available
            if is_available():
                self._semantic = SemanticIndex(model_dir=model_dir)
                logger.info("Kore: nivel semántico activado (ONNX)")
            else:
                logger.info("Kore: nivel semántico no disponible (pip install kore[semantic])")
        except ImportError:
            logger.debug("Kore: ONNX no instalado, nivel semántico desactivado")

    # ── API PRINCIPAL ─────────────────────────────────────────────────────

    def ask(self, query: str, session_id: str | None = None) -> Optional[KoreResult]:
        """
        Pregunta al motor de conocimiento.

        Args:
            query: La pregunta del usuario.
            session_id: ID de sesión para memoria conversacional.
                        Si se pasa, Kore enriquece queries vagas con contexto
                        y registra cada turno automáticamente.

        Recorre los 4 niveles de retrieval en orden.
        Devuelve KoreResult si encuentra algo con score >= min_score.
        Devuelve None si no sabe la respuesta (miss).
        """
        min_score = self._config.min_score

        # ── Memoria: enriquecer query con contexto de sesión ─────────
        original_query = query
        if session_id:
            query = self._memory.enrich(query, session_id)
            if query != original_query:
                logger.debug(f"Kore memory: '{original_query}' → '{query}'")

        result = self._search(query, min_score)

        # ── Memoria: registrar turno ──────────────────────────────────
        self._record_to_memory(session_id, original_query, result)

        return result

    def _search(self, query: str, min_score: float) -> Optional[KoreResult]:
        """Busca en los 4 niveles de retrieval."""

        # ── Nivel 1: Hash exacto ─────────────────────────────────────
        q_norm = normalize(query)
        q_hash = hash_text(q_norm)

        entry = self._store.get_by_id(q_hash)
        if entry:
            self._store.incr_stat("hits_exact")
            return KoreResult(
                answer=entry['answer'],
                score=1.0,
                level=MatchLevel.EXACT,
                source=entry.get('source', ''),
                category=entry.get('category', ''),
                entry_id=q_hash,
            )

        # ── Nivel 2: Keywords + categoría ─────────────────────────────
        if self._config.categories:
            category, keywords = extract_keywords(query, self._config.categories)
            if keywords:
                candidates = self._store.find_by_keywords(category, keywords)
                if candidates:
                    best_id, best_count = candidates[0]
                    score = best_count / len(keywords)
                    if score >= min_score:
                        entry = self._store.get_by_id(best_id)
                        if entry:
                            self._store.incr_stat("hits_keywords")
                            return KoreResult(
                                answer=entry['answer'],
                                score=round(score, 2),
                                level=MatchLevel.KEYWORDS,
                                source=entry.get('source', ''),
                                category=entry.get('category', ''),
                                entry_id=best_id,
                            )

        # ── Nivel 3: BM25 ────────────────────────────────────────────
        # BM25 es más permisivo, usar umbral más alto para evitar falsos positivos
        bm25_min = max(min_score, 0.7)
        if self._bm25.size > 0:
            q_tokens = tokenize(query)
            if q_tokens:
                results = self._bm25.search(q_tokens, top_k=1, min_score=bm25_min)
                if results:
                    best_id, score = results[0]
                    entry = self._store.get_by_id(best_id)
                    if entry:
                        self._store.incr_stat("hits_bm25")
                        return KoreResult(
                            answer=entry['answer'],
                            score=round(score, 2),
                            level=MatchLevel.BM25,
                            source=entry.get('source', ''),
                            category=entry.get('category', ''),
                            entry_id=best_id,
                        )

        # ── Nivel 4: Semántico ────────────────────────────────────────
        if self._semantic is not None:
            results = self._semantic.search(query, top_k=1, min_score=max(min_score, 0.5))
            if results:
                best_id, score = results[0]
                entry = self._store.get_by_id(best_id)
                if entry:
                    self._store.incr_stat("hits_semantic")
                    return KoreResult(
                        answer=entry['answer'],
                        score=round(score, 2),
                        level=MatchLevel.SEMANTIC,
                        source=entry.get('source', ''),
                        category=entry.get('category', ''),
                        entry_id=best_id,
                    )

        # ── Miss ──────────────────────────────────────────────────────
        self._store.incr_stat("misses")
        return None

    def _record_to_memory(self, session_id: str | None, query: str,
                          result: KoreResult | None):
        """Registra el turno en la memoria de sesión."""
        if not session_id or result is None:
            return
        keywords = []
        if self._config.categories:
            _, keywords = extract_keywords(query, self._config.categories)
        self._memory.record(
            session_id=session_id,
            query=query,
            answer=result.answer[:500],
            category=result.category,
            keywords=keywords,
        )

    # ── INGESTA ───────────────────────────────────────────────────────────

    def ingest(self, source: str | Path | list[dict], **kwargs) -> dict:
        """
        Ingesta conocimiento desde cualquier fuente.

        Acepta:
          - str/Path terminado en .jsonl → carga JSONL
          - str/Path terminado en .csv/.tsv → carga CSV
          - list[dict] → carga directa
          - str/Path terminado en .json → carga JSON array

        Returns:
            {"loaded": int, "errors": int, "total": int}
        """
        if isinstance(source, list):
            entries = load_entries(source)
        else:
            source = Path(source)
            if source.suffix == '.jsonl':
                entries = load_jsonl(source, **kwargs)
            elif source.suffix in ('.csv', '.tsv'):
                delimiter = '\t' if source.suffix == '.tsv' else ','
                entries = load_csv(source, delimiter=delimiter, **kwargs)
            elif source.suffix == '.json':
                with open(source, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                entries = load_entries(data if isinstance(data, list) else [data])
            else:
                raise ValueError(f"Formato no soportado: {source.suffix}")

        return self._store_entries(entries)

    def add(self, query: str, answer: str, source: str = "",
            category: str | None = None, metadata: dict | None = None) -> bool:
        """Añade una entrada individual."""
        if category is None and self._config.categories:
            category, _ = extract_keywords(query + " " + answer, self._config.categories)
        category = category or "general"

        entry = KoreEntry(
            query=query, answer=answer, source=source,
            category=category, metadata=metadata or {},
        )
        return self._store_single(entry)

    # ── FEEDBACK / APRENDIZAJE ────────────────────────────────────────────

    def feedback(self, query: str, answer: str, positive: bool = True):
        """
        Registra feedback del usuario.
        Si es positivo y learn_from_feedback está activo, se incorpora al conocimiento.
        """
        category = "general"
        if self._config.categories:
            category, _ = extract_keywords(query, self._config.categories)

        self._store.save_feedback(query, answer, positive,
                                  source="feedback", category=category)

        if positive and self._config.learn_from_feedback:
            self.add(query, answer, source="learned", category=category)
            self._store.incr_stat("learned")

    def learn(self, min_feedback: int = 2) -> int:
        """
        Incorpora feedback positivo acumulado al conocimiento.
        Solo entradas con >= min_feedback votos positivos.
        Returns: número de entries aprendidas.
        """
        positive = self._store.get_positive_feedback(min_count=min_feedback)
        count = 0
        for item in positive:
            added = self.add(
                query=item['query'],
                answer=item['answer'],
                source="learned",
                category=item.get('category', 'general'),
            )
            if added:
                count += 1
        return count

    # ── MEMORIA ───────────────────────────────────────────────────────────

    def context(self, session_id: str) -> dict:
        """Contexto actual de una sesión (topic, entidades, turnos)."""
        return self._memory.get_context(session_id)

    def history(self, session_id: str, last_n: int = 5) -> list[dict]:
        """Últimos N turnos de una sesión."""
        return self._memory.get_history(session_id, last_n)

    def forget(self, session_id: str):
        """Olvida una sesión."""
        self._memory.clear_session(session_id)

    # ── ESTADÍSTICAS ──────────────────────────────────────────────────────

    def stats(self) -> KoreStats:
        """Estadísticas del motor."""
        return self._store.get_stats()

    # ── CONFIGURACIÓN ─────────────────────────────────────────────────────

    def configure(self, config: DomainConfig | dict):
        """Actualiza la configuración del dominio."""
        if isinstance(config, dict):
            config = DomainConfig.from_dict(config)
        self._config = config

    # ── LIMPIEZA ──────────────────────────────────────────────────────────

    def clear(self):
        """Limpia todo el conocimiento."""
        self._store.clear()
        self._bm25.clear()
        if self._semantic:
            self._semantic.clear()

    def close(self):
        """Cierra conexiones."""
        self._store.close()

    # ── INTERNOS ──────────────────────────────────────────────────────────

    def _store_entries(self, entries: list[KoreEntry]) -> dict:
        """Almacena entries y actualiza índices."""
        items = []
        for entry in entries:
            q_norm = normalize(entry.query)
            q_hash = hash_text(q_norm)

            if self._config.categories:
                category, keywords = extract_keywords(
                    entry.query + " " + entry.answer,
                    self._config.categories,
                )
            else:
                category = entry.category or "general"
                keywords = []

            items.append((q_hash, entry, q_norm, keywords, category))

        loaded, errors = self._store.put_bulk(items)

        # Rebuild BM25
        self._rebuild_bm25()

        # Semantic index
        if self._semantic:
            for q_hash, entry, _, _, _ in items:
                self._semantic.add(q_hash, entry.query)

        logger.info(f"Kore [{self.domain}]: {loaded} entries ingestadas, {errors} errores")
        return {"loaded": loaded, "errors": errors, "total": len(entries)}

    def _store_single(self, entry: KoreEntry) -> bool:
        """Almacena una entry individual."""
        q_norm = normalize(entry.query)
        q_hash = hash_text(q_norm)

        if self._config.categories:
            category, keywords = extract_keywords(
                entry.query + " " + entry.answer,
                self._config.categories,
            )
        else:
            category = entry.category or "general"
            keywords = []

        ok = self._store.put(q_hash, entry, q_norm, keywords, category)

        if ok:
            # Actualizar BM25
            tokens = tokenize(entry.query + " " + entry.answer)
            self._bm25.add(q_hash, tokens)
            self._bm25.build()

            # Semantic
            if self._semantic:
                self._semantic.add(q_hash, entry.query)

        return ok

    def _rebuild_bm25(self):
        """Reconstruye el índice BM25 desde el store."""
        self._bm25.clear()
        entries = self._store.get_all_entries()
        for entry in entries:
            tokens = tokenize(entry['query'] + " " + entry.get('answer', ''))
            self._bm25.add(entry['id'], tokens)
        self._bm25.build()

    def __repr__(self) -> str:
        count = self._store.count()
        levels = "3" if self._semantic is None else "4"
        return f"Kore(domain={self.domain!r}, entries={count}, levels={levels})"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
