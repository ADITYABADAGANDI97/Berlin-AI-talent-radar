"""
Numpy-backed vector store for cosine similarity search.

Stores embeddings as pre-normalized numpy arrays (.npz) and chunk
metadata/text as JSON (.json), side by side with parallel indexing.
Supports filtered search by source_type and arbitrary ChunkMetadata
fields.

Usage::

    from src.storage.numpy_store import NumpyVectorStore
    store = NumpyVectorStore()
    store.save_job_chunks(job_chunks)
    store.save_regulation_chunks(reg_chunks)
    store.load()
    results = store.search(query_embedding, top_k=6)
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.models import Chunk, ChunkMetadata, SearchResult
from src.utils.Io import PROJECT_ROOT, save_json, load_json
from src.utils.logger import get_logger

logger = get_logger("storage.NumpyVectorStore")


class NumpyVectorStore:
    """
    Numpy-backed vector store with cosine similarity search and
    metadata filtering.

    Storage format per collection:
    - ``{name}.npz``: numpy float32 matrix (N x 1536), pre-normalized
    - ``{name}_meta.json``: chunk text + metadata (embedding stripped)
    """

    def __init__(self) -> None:
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh)

        np_cfg = settings.get("vector_store", {}).get("numpy", {})
        self._save_dir = PROJECT_ROOT / np_cfg.get("save_dir", "data/embeddings")
        self._jobs_npz = PROJECT_ROOT / np_cfg.get(
            "jobs_file", "data/embeddings/job_chunks.npz"
        )
        self._regs_npz = PROJECT_ROOT / np_cfg.get(
            "regulations_file", "data/embeddings/regulation_chunks.npz"
        )

        # In-memory indices
        self._job_embeddings: np.ndarray | None = None
        self._job_chunks: list[Chunk] = []
        self._reg_embeddings: np.ndarray | None = None
        self._reg_chunks: list[Chunk] = []

        logger.info("NumpyVectorStore initialized: %s", self._save_dir)

    @property
    def total_chunks(self) -> int:
        """Total chunks loaded in memory across all collections."""
        return len(self._job_chunks) + len(self._reg_chunks)

    @property
    def job_chunk_count(self) -> int:
        return len(self._job_chunks)

    @property
    def regulation_chunk_count(self) -> int:
        return len(self._reg_chunks)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_job_chunks(self, chunks: list[Chunk]) -> None:
        """
        Persist job chunks to disk. Filters out chunks without embeddings.

        Args:
            chunks: Job posting chunks with embeddings populated.
        """
        self._save_collection(chunks, self._jobs_npz, "job")

    def save_regulation_chunks(self, chunks: list[Chunk]) -> None:
        """
        Persist EU AI Act regulation chunks to disk.

        Args:
            chunks: Regulation chunks with embeddings populated.
        """
        self._save_collection(chunks, self._regs_npz, "regulation")

    def _save_collection(
        self, chunks: list[Chunk], npz_path: Path, label: str,
    ) -> None:
        """Serialize a collection: embeddings as .npz, metadata as .json."""
        # Filter out chunks without embeddings
        valid = [c for c in chunks if c.embedding is not None]
        skipped = len(chunks) - len(valid)
        if skipped > 0:
            logger.warning(
                "%d %s chunks had no embedding and were excluded", skipped, label,
            )

        if not valid:
            logger.warning("No valid %s chunks to save", label)
            return

        # Build embedding matrix and normalize
        matrix = np.array([c.embedding for c in valid], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        matrix = matrix / norms

        # Save embeddings
        npz_path = Path(npz_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(npz_path), embeddings=matrix)

        # Save chunk metadata (strip embedding to avoid duplication)
        meta_path = npz_path.with_name(npz_path.stem + "_meta.json")
        meta_list = []
        for c in valid:
            d = c.model_dump()
            d.pop("embedding", None)
            meta_list.append(d)
        save_json(meta_list, meta_path)

        logger.info(
            "Saved %d %s chunks: %s (%s)",
            len(valid), label, npz_path, meta_path,
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load all chunks and embeddings from disk into memory.

        Safe to call if files don't exist — starts with empty collections.
        """
        self._job_embeddings, self._job_chunks = self._load_collection(
            self._jobs_npz, "job",
        )
        self._reg_embeddings, self._reg_chunks = self._load_collection(
            self._regs_npz, "regulation",
        )
        logger.info(
            "Vector store loaded: %d job chunks, %d regulation chunks",
            len(self._job_chunks), len(self._reg_chunks),
        )

    def _load_collection(
        self, npz_path: Path, label: str,
    ) -> tuple[np.ndarray | None, list[Chunk]]:
        """Deserialize a collection from disk."""
        npz_path = Path(npz_path)
        meta_path = npz_path.with_name(npz_path.stem + "_meta.json")

        if not npz_path.exists() or not meta_path.exists():
            logger.debug("No %s collection found at %s", label, npz_path)
            return None, []

        # Load embeddings
        data = np.load(str(npz_path))
        matrix = data["embeddings"]

        # Load metadata and reconstruct Chunks
        meta_list = load_json(meta_path)
        chunks: list[Chunk] = []
        for i, d in enumerate(meta_list):
            # Re-attach embedding from numpy row
            d["embedding"] = matrix[i].tolist()
            chunks.append(Chunk(**d))

        logger.info("Loaded %d %s chunks from %s", len(chunks), label, npz_path)
        return matrix, chunks

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        source_type: str | None = None,
        top_k: int = 6,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Find the top_k most similar chunks by cosine similarity.

        Args:
            query_embedding: 1536-dim query vector.
            source_type: Filter to "job_posting" or "eu_ai_act".
                If None, searches both collections.
            top_k: Number of results to return.
            filters: Optional metadata filters, e.g.
                {"governance_gap": True, "seniority": "senior"}.

        Returns:
            List of SearchResult sorted by similarity descending.
        """
        results: list[SearchResult] = []

        # Normalize query
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Search job collection
        if source_type in (None, "job_posting"):
            results.extend(
                self._search_collection(
                    q, self._job_embeddings, self._job_chunks, filters,
                )
            )

        # Search regulation collection
        if source_type in (None, "eu_ai_act"):
            results.extend(
                self._search_collection(
                    q, self._reg_embeddings, self._reg_chunks, filters,
                )
            )

        # Sort by similarity and return top_k
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def _search_collection(
        self,
        query: np.ndarray,
        embeddings: np.ndarray | None,
        chunks: list[Chunk],
        filters: dict[str, Any] | None,
    ) -> list[SearchResult]:
        """Search a single collection with optional filtering."""
        if embeddings is None or len(chunks) == 0:
            return []

        # Apply metadata filters
        if filters:
            indices = self._apply_filters(chunks, filters)
            if not indices:
                return []
            filtered_embeddings = embeddings[indices]
            filtered_chunks = [chunks[i] for i in indices]
        else:
            filtered_embeddings = embeddings
            filtered_chunks = chunks

        # Cosine similarity (embeddings are pre-normalized, so dot product suffices)
        scores = filtered_embeddings @ query

        results = []
        for i, score in enumerate(scores):
            sim = float(max(0.0, min(1.0, score)))
            results.append(SearchResult(
                chunk=filtered_chunks[i],
                similarity=sim,
            ))
        return results

    def _apply_filters(
        self, chunks: list[Chunk], filters: dict[str, Any],
    ) -> list[int]:
        """Return indices of chunks matching all filter criteria."""
        indices: list[int] = []
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata
            match = True
            for key, value in filters.items():
                meta_val = getattr(meta, key, None)
                if isinstance(meta_val, list) and not isinstance(value, list):
                    # Check if value is in the list
                    if value not in meta_val:
                        match = False
                        break
                elif meta_val != value:
                    match = False
                    break
            if match:
                indices.append(i)
        return indices
