"""Hybrid search over the course chunks: dense embeddings plus BM25, fused by rank."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .schema import Chunk

DEFAULT_EMBED_MODEL = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60

_WORD = re.compile(r"[a-z0-9_]+")
_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "for", "on", "with", "as",
         "by", "that", "this", "it", "be", "what", "how", "why", "do", "does", "can", "i", "you"}


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP]


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-9, None)


class FastEmbedder:
    """Small ONNX embedding model (no PyTorch needed). Loaded on first use."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name,
                                        cache_dir=os.getenv("FASTEMBED_CACHE_PATH") or None)
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.array(list(self._load().embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        vec = np.array(list(self._load().embed([QUERY_PREFIX + text])), dtype=np.float32)[0]
        return _normalise(vec)


@dataclass
class Hit:
    chunk: Chunk
    dense: float      # cosine similarity with the query (used to decide when to refuse)
    fused: float      # reciprocal-rank-fusion score (used for ordering)


def load_chunks(path: Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as fh:
        return [Chunk.model_validate_json(line) for line in fh if line.strip()]


class Index:
    def __init__(self, chunks: list[Chunk], matrix: np.ndarray, embedder):
        if len(chunks) != len(matrix):
            raise ValueError("chunks and embeddings are out of step; rebuild the index")
        self.chunks, self.matrix, self.embedder = chunks, matrix, embedder
        self._bm25 = BM25Okapi([tokenize(c.searchable_text) or ["_"] for c in chunks])

    @classmethod
    def build(cls, chunks: list[Chunk], embedder, batch: int = 64) -> "Index":
        parts = [embedder.embed_documents([c.searchable_text[:2000] for c in chunks[i:i + batch]])
                 for i in range(0, len(chunks), batch)]
        return cls(chunks, np.vstack(parts).astype(np.float32), embedder)

    def save(self, directory: Path) -> None:
        np.save(directory / "embeddings.npy", self.matrix)

    @classmethod
    def load(cls, directory: Path, embedder) -> "Index":
        return cls(load_chunks(directory / "chunks.jsonl"), np.load(directory / "embeddings.npy"), embedder)

    def search(self, query: str, k: int = 6, pool: int = 30) -> list[Hit]:
        q = self.embedder.embed_query(query)
        dense = self.matrix @ q
        sparse = self._bm25.get_scores(tokenize(query) or ["_"])
        fused: dict[int, float] = {}
        for rank, i in enumerate(np.argsort(-dense)[:pool]):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, i in enumerate(np.argsort(-sparse)[:pool]):
            if sparse[i] > 0:
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.chunks[i], float(dense[i]), fused[i]) for i in top]