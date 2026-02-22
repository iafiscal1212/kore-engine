"""
Kore — Semantic Retrieval (Nivel 4)
════════════════════════════════════
Mini-transformer ONNX para embeddings semánticos.
Activar con: pip install kore-engine[semantic]

Auto-descarga el modelo en la primera ejecución.
Sin PyTorch. Sin GPU. ~35MB total.

    k = Kore("mi_dominio", semantic=True)

IAFiscal © 2026
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Detección de dependencias opcionales
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
# Configuración del modelo
# ─────────────────────────────────────────────────────────────────────────────

_CACHE_DIR = Path(os.environ.get(
    "KORE_CACHE_DIR",
    Path.home() / ".cache" / "kore" / "models"
))

# all-MiniLM-L6-v2 — embeddings multilingüe, rápido, ligero
_HF_REPO = "sentence-transformers/all-MiniLM-L6-v2"
_HF_BASE = f"https://huggingface.co/{_HF_REPO}/resolve/main"

_MODEL_FILES = {
    "model.onnx": {
        "url": f"{_HF_BASE}/onnx/model.onnx",
        "size_mb": 23,
    },
    "tokenizer.json": {
        "url": f"{_HF_BASE}/tokenizer.json",
        "size_mb": 0.7,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Auto-descarga del modelo
# ─────────────────────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, desc: str = ""):
    """Descarga un archivo con progreso simple."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "kore-engine"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 64 * 1024  # 64KB

            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        pct = downloaded * 100 // total
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  Descargando {desc}... {mb:.1f}MB ({pct}%)", end="", flush=True)

            if total > 0:
                print()  # Nueva línea al terminar

        # Mover de tmp a destino (atómico en mismo filesystem)
        shutil.move(str(tmp), str(dest))
        return True

    except (urllib.error.URLError, OSError) as e:
        tmp.unlink(missing_ok=True)
        logger.error(f"Error descargando {url}: {e}")
        raise RuntimeError(
            f"No se pudo descargar el modelo semántico.\n"
            f"  URL: {url}\n"
            f"  Error: {e}\n"
            f"  Comprueba tu conexión a internet."
        ) from e


def _ensure_model(model_dir: Path | None = None) -> Path:
    """
    Asegura que los archivos del modelo existen.
    Si no están, los descarga automáticamente de HuggingFace.
    """
    model_path = model_dir or _CACHE_DIR

    onnx_path = model_path / "model.onnx"
    tokenizer_path = model_path / "tokenizer.json"

    if onnx_path.exists() and tokenizer_path.exists():
        return model_path

    # Descargar lo que falte
    model_path.mkdir(parents=True, exist_ok=True)
    print(f"  Kore Semantic: descargando modelo ({_HF_REPO})...")
    print(f"  Destino: {model_path}")

    for filename, info in _MODEL_FILES.items():
        filepath = model_path / filename
        if not filepath.exists():
            _download_file(info["url"], filepath, desc=f"{filename} (~{info['size_mb']}MB)")

    # Verificar que se descargaron bien
    if not onnx_path.exists():
        raise RuntimeError(f"Modelo ONNX no encontrado tras descarga: {onnx_path}")
    if not tokenizer_path.exists():
        raise RuntimeError(f"Tokenizer no encontrado tras descarga: {tokenizer_path}")

    print(f"  Kore Semantic: modelo listo")
    return model_path


# ─────────────────────────────────────────────────────────────────────────────
# Índice Semántico
# ─────────────────────────────────────────────────────────────────────────────

class SemanticIndex:
    """
    Índice semántico basado en embeddings ONNX.

    Calcula similitud coseno entre la query y las entries.
    Mantiene embeddings precalculados en memoria.

    Auto-descarga el modelo en la primera ejecución.
    """

    def __init__(self, model_dir: str | Path | None = None):
        if not is_available():
            raise ImportError(
                "Nivel semántico requiere: pip install kore-engine[semantic]\n"
                "  (instala onnxruntime + tokenizers, ~35MB)\n"
                "  Sin PyTorch. Sin GPU. Corre en cualquier CPU."
            )

        # Auto-descargar modelo si no existe
        model_path = _ensure_model(
            Path(model_dir) if model_dir else None
        )

        # Cargar sesión ONNX
        onnx_path = model_path / "model.onnx"
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=['CPUExecutionProvider']
        )

        # Cargar tokenizer
        tokenizer_path = model_path / "tokenizer.json"
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_padding(length=128)
        self._tokenizer.enable_truncation(max_length=128)

        # Embeddings precalculados: doc_id -> embedding (numpy array)
        self._embeddings: dict[str, list[float]] = {}

        # Dimensión del embedding (se detecta en el primer encode)
        self._dim: int = 0

        logger.info("Kore Semantic: modelo ONNX cargado desde %s", model_path)

    def add(self, doc_id: str, text: str):
        """Calcula y almacena el embedding de un texto."""
        emb = self._encode(text)
        self._embeddings[doc_id] = emb

    def add_batch(self, items: list[tuple[str, str]]):
        """Calcula embeddings en batch (más eficiente)."""
        for doc_id, text in items:
            self._embeddings[doc_id] = self._encode(text)

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
        """Elimina un embedding."""
        self._embeddings.pop(doc_id, None)

    def clear(self):
        """Limpia todos los embeddings."""
        self._embeddings.clear()

    @property
    def size(self) -> int:
        """Número de embeddings almacenados."""
        return len(self._embeddings)

    def _encode(self, text: str) -> list[float]:
        """Genera embedding para un texto usando ONNX."""
        import numpy as np

        encoded = self._tokenizer.encode(text)
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

        if not self._dim:
            self._dim = len(embedding)

        return embedding

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Similitud coseno entre dos vectores."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
