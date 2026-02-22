"""
Kore — Semantic Retrieval (Nivel 4, opcional)
══════════════════════════════════════════════
Mini-transformer ONNX para embeddings semánticos.
Solo se activa con: pip install kore[semantic]

Modelo: all-MiniLM-L6-v2 (cuantizado, ~20MB)
Runtime: ONNX Runtime (~15MB)
Total: ~35MB vs ~2GB de PyTorch+transformers

IAFiscal © 2026
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Detección de ONNX Runtime
# ─────────────────────────────────────────────────────────────────────────────

_ONNX_AVAILABLE = False
_TOKENIZERS_AVAILABLE = False

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    pass

try:
    from tokenizers import Tokenizer
    _TOKENIZERS_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """¿Está disponible el nivel semántico?"""
    return _ONNX_AVAILABLE and _TOKENIZERS_AVAILABLE


# ─────────────────────────────────────────────────────────────────────────────
# Modelo ONNX
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_DIR = Path(__file__).parent.parent / "models"
_MODEL_FILE = "minilm_l6_v2_quantized.onnx"
_TOKENIZER_FILE = "tokenizer.json"


class SemanticIndex:
    """
    Índice semántico basado en embeddings ONNX.

    Calcula similitud coseno entre la query y las entries.
    Mantiene embeddings precalculados en memoria.
    """

    def __init__(self, model_dir: str | Path | None = None):
        if not is_available():
            raise ImportError(
                "Semantic search requiere: pip install kore[semantic]\n"
                "  (instala onnxruntime + tokenizers, ~35MB total)"
            )

        model_path = Path(model_dir) if model_dir else _MODEL_DIR

        onnx_path = model_path / _MODEL_FILE
        tokenizer_path = model_path / _TOKENIZER_FILE

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Modelo ONNX no encontrado en {onnx_path}.\n"
                f"Descárgalo con: kore download-model"
            )

        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=['CPUExecutionProvider']
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_padding(length=128)
        self._tokenizer.enable_truncation(max_length=128)

        # Embeddings precalculados: doc_id -> embedding (list[float])
        self._embeddings: dict[str, list[float]] = {}
        logger.info("Kore Semantic: modelo ONNX cargado")

    def add(self, doc_id: str, text: str):
        """Calcula y almacena el embedding de un texto."""
        emb = self._encode(text)
        self._embeddings[doc_id] = emb

    def search(self, query: str, top_k: int = 5,
               min_score: float = 0.5) -> list[tuple[str, float]]:
        """Busca por similitud coseno."""
        if not self._embeddings:
            return []

        q_emb = self._encode(query)
        scores = []

        for doc_id, doc_emb in self._embeddings.items():
            sim = self._cosine_similarity(q_emb, doc_emb)
            if sim >= min_score:
                scores.append((doc_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def remove(self, doc_id: str):
        self._embeddings.pop(doc_id, None)

    def clear(self):
        self._embeddings.clear()

    @property
    def size(self) -> int:
        return len(self._embeddings)

    def _encode(self, text: str) -> list[float]:
        """Genera embedding para un texto."""
        encoded = self._tokenizer.encode(text)
        import numpy as np
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        # Mean pooling
        token_embeddings = outputs[0][0]  # (seq_len, hidden_dim)
        mask = attention_mask[0]
        masked = token_embeddings * mask[:, None]
        summed = masked.sum(axis=0)
        count = mask.sum()
        embedding = (summed / count).tolist()
        return embedding

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
