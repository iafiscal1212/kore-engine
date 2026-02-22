"""
Kore — Ingestores de datos
═══════════════════════════
Carga conocimiento desde distintos formatos.
Soporta: JSONL, CSV, lista de dicts, ChatML.

IAFiscal © 2026
"""

from __future__ import annotations

import csv
import json
import re
import logging
from pathlib import Path
from typing import Iterator

from kore.models import KoreEntry

logger = logging.getLogger(__name__)


def load_jsonl(path: str | Path) -> list[KoreEntry]:
    """
    Carga entries desde un archivo JSONL.

    Formatos soportados por línea:
      - {"query": "...", "answer": "...", "source": "...", "category": "..."}
      - {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
      - {"text": "<|im_start|>user\\n..."}  (ChatML)
      - {"question": "...", "answer": "..."}  (alternativo)
    """
    entries = []
    seen = set()
    path = Path(path)

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entry = _parse_entry(data)
                if entry and entry.query not in seen:
                    entries.append(entry)
                    seen.add(entry.query)
            except (json.JSONDecodeError, Exception) as e:
                logger.debug(f"Línea {i+1} ignorada: {e}")

    logger.info(f"Cargadas {len(entries)} entries desde {path.name}")
    return entries


def load_csv(path: str | Path, query_col: str = "query",
             answer_col: str = "answer", source_col: str = "source",
             category_col: str = "category",
             delimiter: str = ",") -> list[KoreEntry]:
    """Carga entries desde CSV/TSV."""
    entries = []
    seen = set()
    path = Path(path)

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            query = row.get(query_col, "").strip()
            answer = row.get(answer_col, "").strip()
            if query and answer and query not in seen:
                entries.append(KoreEntry(
                    query=query,
                    answer=answer,
                    source=row.get(source_col, ""),
                    category=row.get(category_col, "general"),
                ))
                seen.add(query)

    logger.info(f"Cargadas {len(entries)} entries desde {path.name}")
    return entries


def load_entries(data: list[dict]) -> list[KoreEntry]:
    """Carga entries desde una lista de dicts."""
    entries = []
    seen = set()
    for item in data:
        entry = _parse_entry(item)
        if entry and entry.query not in seen:
            entries.append(entry)
            seen.add(entry.query)
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Parsers internos
# ─────────────────────────────────────────────────────────────────────────────

def _parse_entry(data: dict) -> KoreEntry | None:
    """Parsea una entry desde distintos formatos."""
    source = data.get("source", "")
    category = data.get("category", data.get("categoria", "general"))
    metadata = {k: v for k, v in data.items()
                if k not in ("query", "answer", "question", "respuesta",
                             "source", "category", "categoria",
                             "conversations", "text", "metadata")}
    if "metadata" in data and isinstance(data["metadata"], dict):
        metadata.update(data["metadata"])

    # Formato 1: query/answer directo
    query = data.get("query", data.get("question", "")).strip()
    answer = data.get("answer", data.get("respuesta", "")).strip()
    if query and answer:
        return KoreEntry(query=query, answer=answer, source=source,
                         category=category, metadata=metadata)

    # Formato 2: conversations
    if "conversations" in data:
        convs = data["conversations"]
        user_msg = ""
        assistant_msg = ""
        for msg in convs:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user" and not user_msg:
                user_msg = content.strip()
            elif role == "assistant" and not assistant_msg:
                assistant_msg = content.strip()
        if user_msg and assistant_msg:
            return KoreEntry(query=user_msg, answer=assistant_msg,
                             source=source, category=category, metadata=metadata)

    # Formato 3: ChatML
    if "text" in data:
        text = data["text"]
        user_msg = _extract_chatml(text, "user")
        assistant_msg = _extract_chatml(text, "assistant")
        if user_msg and assistant_msg:
            return KoreEntry(query=user_msg, answer=assistant_msg,
                             source=source, category=category, metadata=metadata)

    return None


def _extract_chatml(text: str, role: str) -> str | None:
    """Extrae contenido de un rol en formato ChatML."""
    pattern = rf'<\|im_start\|>{role}\n(.*?)<\|im_end\|>'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None
