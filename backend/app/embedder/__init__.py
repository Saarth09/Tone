from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self) -> None:
        self.settings = get_settings()
        Path(self.settings.chroma_path).mkdir(parents=True, exist_ok=True)
        logger.info("Loading embedding model %s", self.settings.embedding_model)
        self.model = SentenceTransformer(self.settings.embedding_model)
        self.client = chromadb.PersistentClient(path=self.settings.chroma_path)
        self.collection = self.client.get_or_create_collection(
            name="tone_embeddings",
            metadata={"hnsw:space": "cosine"},
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def store(
        self,
        sample_id: int,
        text: str,
        *,
        probe_id: str,
        category: str,
        is_baseline: bool,
        created_at: str,
        vector: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if vector is None:
            vector = self.embed([text])[0]
        self.collection.upsert(
            ids=[f"sample_{sample_id}"],
            embeddings=[vector.tolist()],
            documents=[text],
            metadatas=[
                {
                    "sample_id": int(sample_id),
                    "probe_id": probe_id,
                    "category": category,
                    "is_baseline": int(bool(is_baseline)),
                    "created_at": created_at,
                }
            ],
        )
        return vector

    def get_baseline_vectors(self, category: Optional[str] = None) -> np.ndarray:
        where: dict = {"is_baseline": 1}
        if category:
            where = {"$and": [{"is_baseline": 1}, {"category": category}]}
        try:
            result = self.collection.get(where=where, include=["embeddings"])
        except Exception:
            return np.empty((0, 384), dtype=np.float32)
        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return np.empty((0, 384), dtype=np.float32)
        return np.asarray(embeddings, dtype=np.float32)

    def get_recent_live_vectors(
        self, category: Optional[str] = None, limit: int = 50
    ) -> np.ndarray:
        where: dict = {"is_baseline": 0}
        if category:
            where = {"$and": [{"is_baseline": 0}, {"category": category}]}
        try:
            result = self.collection.get(where=where, include=["embeddings", "metadatas"])
        except Exception:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = result.get("embeddings")
        metadatas = result.get("metadatas") or []
        if embeddings is None or len(embeddings) == 0:
            return np.empty((0, 384), dtype=np.float32)

        pairs = list(zip(list(embeddings), metadatas))
        pairs.sort(key=lambda x: x[1].get("created_at", ""), reverse=True)
        selected = [e for e, _ in pairs[:limit]]
        return np.asarray(selected, dtype=np.float32)

    def nearest_baseline(
        self, vector: np.ndarray, probe_id: str, k: int = 5
    ) -> tuple[list[float], list[str]]:
        try:
            result = self.collection.query(
                query_embeddings=[vector.tolist()],
                n_results=k,
                where={"$and": [{"is_baseline": 1}, {"probe_id": probe_id}]},
                include=["distances", "documents"],
            )
        except Exception:
            return [], []

        distances = (result.get("distances") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        # Chroma cosine distance = 1 - cosine_similarity
        similarities = [1.0 - float(d) for d in distances]
        return similarities, list(documents or [])

    def all_baseline_docs(self, probe_id: str, limit: int = 5) -> list[str]:
        try:
            result = self.collection.get(
                where={"$and": [{"is_baseline": 1}, {"probe_id": probe_id}]},
                include=["documents", "metadatas"],
                limit=limit,
            )
        except Exception:
            return []
        docs = result.get("documents")
        return list(docs or [])
