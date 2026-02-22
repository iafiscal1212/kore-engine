"""
Kore — SQLite Storage
═════════════════════
Almacén persistente basado en SQLite.
Cero dependencias: SQLite viene con Python.

Esquema:
  entries    — conocimiento (query, answer, category, source, metadata)
  keywords   — índice invertido de keywords por categoría
  stats      — contadores de hits/misses
  feedback   — respuestas validadas por usuarios (aprendizaje)

IAFiscal © 2026
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from kore.models import KoreEntry, KoreStats


class SQLiteStore:
    """Almacén SQLite thread-safe para Kore."""

    def __init__(self, path: str | Path = ":memory:"):
        self._path = str(path)
        self._local = threading.local()
        self._init_schema()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_schema(self):
        conn = self._conn
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                query_normalized TEXT NOT NULL,
                answer TEXT NOT NULL,
                source TEXT DEFAULT '',
                category TEXT DEFAULT 'general',
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_entries_hash ON entries(id);
            CREATE INDEX IF NOT EXISTS idx_entries_category ON entries(category);

            CREATE TABLE IF NOT EXISTS keywords (
                keyword TEXT NOT NULL,
                category TEXT NOT NULL,
                entry_id TEXT NOT NULL,
                FOREIGN KEY (entry_id) REFERENCES entries(id)
            );

            CREATE INDEX IF NOT EXISTS idx_kw_cat ON keywords(category, keyword);
            CREATE INDEX IF NOT EXISTS idx_kw_entry ON keywords(entry_id);

            CREATE TABLE IF NOT EXISTS stats (
                key TEXT PRIMARY KEY,
                value INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                source TEXT DEFAULT 'feedback',
                category TEXT DEFAULT 'general',
                positive INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Inicializar contadores
            INSERT OR IGNORE INTO stats(key, value) VALUES ('hits_exact', 0);
            INSERT OR IGNORE INTO stats(key, value) VALUES ('hits_keywords', 0);
            INSERT OR IGNORE INTO stats(key, value) VALUES ('hits_bm25', 0);
            INSERT OR IGNORE INTO stats(key, value) VALUES ('hits_semantic', 0);
            INSERT OR IGNORE INTO stats(key, value) VALUES ('misses', 0);
            INSERT OR IGNORE INTO stats(key, value) VALUES ('learned', 0);
        """)
        conn.commit()

    # ── CRUD ──────────────────────────────────────────────────────────────

    def put(self, entry_id: str, entry: KoreEntry, query_normalized: str,
            keywords: list[str], category: str) -> bool:
        """Guarda una entrada con su índice de keywords."""
        try:
            conn = self._conn
            conn.execute(
                """INSERT OR REPLACE INTO entries
                   (id, query, query_normalized, answer, source, category, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (entry_id, entry.query, query_normalized, entry.answer,
                 entry.source, category, json.dumps(entry.metadata))
            )
            # Limpiar keywords anteriores de esta entry
            conn.execute("DELETE FROM keywords WHERE entry_id = ?", (entry_id,))
            # Insertar nuevas keywords
            if keywords:
                conn.executemany(
                    "INSERT INTO keywords (keyword, category, entry_id) VALUES (?, ?, ?)",
                    [(kw, category, entry_id) for kw in keywords]
                )
            conn.commit()
            return True
        except Exception:
            return False

    def get_by_id(self, entry_id: str) -> Optional[dict]:
        """Obtiene una entrada por su hash ID."""
        row = self._conn.execute(
            "SELECT * FROM entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if row:
            return dict(row)
        return None

    def find_by_keywords(self, category: str, keywords: list[str],
                         fallback_all_categories: bool = True) -> list[tuple[str, int]]:
        """
        Busca entries que coincidan con keywords en una categoría.
        Returns: lista de (entry_id, count) ordenada por count desc.
        """
        if not keywords:
            return []

        placeholders = ','.join('?' * len(keywords))
        conn = self._conn

        # Buscar en la categoría específica
        rows = conn.execute(
            f"""SELECT entry_id, COUNT(*) as cnt
                FROM keywords
                WHERE category = ? AND keyword IN ({placeholders})
                GROUP BY entry_id
                ORDER BY cnt DESC
                LIMIT 10""",
            [category] + keywords
        ).fetchall()

        if not rows and fallback_all_categories:
            # Fallback: buscar en todas las categorías
            rows = conn.execute(
                f"""SELECT entry_id, COUNT(*) as cnt
                    FROM keywords
                    WHERE keyword IN ({placeholders})
                    GROUP BY entry_id
                    ORDER BY cnt DESC
                    LIMIT 10""",
                keywords
            ).fetchall()

        return [(row['entry_id'], row['cnt']) for row in rows]

    def get_all_entries(self) -> list[dict]:
        """Obtiene todas las entries (para BM25 index build)."""
        rows = self._conn.execute(
            "SELECT id, query, query_normalized, answer, source, category FROM entries"
        ).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        """Número total de entries."""
        row = self._conn.execute("SELECT COUNT(*) as c FROM entries").fetchone()
        return row['c'] if row else 0

    # ── BULK ──────────────────────────────────────────────────────────────

    def put_bulk(self, items: list[tuple[str, KoreEntry, str, list[str], str]],
                 batch_size: int = 500) -> tuple[int, int]:
        """
        Carga masiva. Cada item: (entry_id, entry, query_normalized, keywords, category).
        Returns: (cargados, errores)
        """
        loaded = 0
        errors = 0
        conn = self._conn

        for i, (eid, entry, qnorm, kws, cat) in enumerate(items):
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO entries
                       (id, query, query_normalized, answer, source, category, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (eid, entry.query, qnorm, entry.answer,
                     entry.source, cat, json.dumps(entry.metadata))
                )
                conn.execute("DELETE FROM keywords WHERE entry_id = ?", (eid,))
                if kws:
                    conn.executemany(
                        "INSERT INTO keywords (keyword, category, entry_id) VALUES (?, ?, ?)",
                        [(kw, cat, eid) for kw in kws]
                    )
                loaded += 1

                if (i + 1) % batch_size == 0:
                    conn.commit()
            except Exception:
                errors += 1

        conn.commit()
        return loaded, errors

    # ── STATS ─────────────────────────────────────────────────────────────

    def incr_stat(self, key: str, amount: int = 1):
        self._conn.execute(
            "UPDATE stats SET value = value + ? WHERE key = ?", (amount, key)
        )
        self._conn.commit()

    def get_stats(self) -> KoreStats:
        rows = self._conn.execute("SELECT key, value FROM stats").fetchall()
        data = {r['key']: r['value'] for r in rows}
        stats = KoreStats(
            total_entries=self.count(),
            hits_exact=data.get('hits_exact', 0),
            hits_keywords=data.get('hits_keywords', 0),
            hits_bm25=data.get('hits_bm25', 0),
            hits_semantic=data.get('hits_semantic', 0),
            misses=data.get('misses', 0),
            learned=data.get('learned', 0),
        )
        return stats

    # ── FEEDBACK (aprendizaje) ────────────────────────────────────────────

    def save_feedback(self, query: str, answer: str, positive: bool,
                      source: str = "feedback", category: str = "general"):
        self._conn.execute(
            """INSERT INTO feedback (query, answer, source, category, positive)
               VALUES (?, ?, ?, ?, ?)""",
            (query, answer, source, category, 1 if positive else 0)
        )
        self._conn.commit()

    def get_positive_feedback(self, min_count: int = 1) -> list[dict]:
        """Obtiene feedback positivo para incorporar al conocimiento."""
        rows = self._conn.execute(
            """SELECT query, answer, source, category, COUNT(*) as cnt
               FROM feedback WHERE positive = 1
               GROUP BY query HAVING cnt >= ?
               ORDER BY cnt DESC""",
            (min_count,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── CLEANUP ───────────────────────────────────────────────────────────

    def clear(self):
        """Limpia toda la base de datos."""
        conn = self._conn
        conn.executescript("""
            DELETE FROM keywords;
            DELETE FROM entries;
            DELETE FROM feedback;
            UPDATE stats SET value = 0;
        """)
        conn.commit()

    def close(self):
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
