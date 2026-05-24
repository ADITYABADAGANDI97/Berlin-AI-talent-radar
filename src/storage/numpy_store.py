"""
Numpy-backed vector store with cosine similarity search.

Two collections live side by side: ``job_postings`` and ``eu_ai_act``.
Each collection persists as two files:

  - ``{name}.npz``        — float32 matrix (N x dim), pre-normalized
  - ``{name}_meta.json``  — chunk text + ChunkMetadata, embedding stripped

Pre-normalizing at save time means cosine similarity is just a dot
product, which keeps the search loop dependency-free and fast enough
for the project's expected scale (~5–10k chunks).

Filters are evaluated against ``ChunkMetadata`` attributes. Scalar
filters match by equality; list-valued metadata (e.g. ``skills``,
``high_risk_domains``) matches when the filter value is contained
in the list. Passing a list as the filter value uses set-intersection
("any of these skills").

Usage::

    from src.storage.numpy_store import NumpyVectorStore
    store = NumpyVectorStore()
    store.save_job_chunks(job_chunks)
    store.save_regulation_chunks(reg_chunks)
    store.load()
    results = store.search(query_vec, top_k=6,
                           filters={"governance_gap": True})
"""

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.models import Chunk, SearchResult
from src.utils.io import PROJECT_ROOT, load_json, save_json
from src.utils.logger import get_logger

logger = get_logger("storage.NumpyVectorStore")


class NumpyVectorStore:
    """Numpy-backed vector store with cosine search and metadata filters."""

    def __init__(self) -> None:
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh)

        np_cfg = settings.get("vector_store", {}).get("numpy", {})
        self._save_dir: Path = PROJECT_ROOT / np_cfg.get(
            "save_dir", "data/embeddings",
        )
        self._jobs_npz: Path = PROJECT_ROOT / np_cfg.get(
            "jobs_file", "data/embeddings/job_chunks.npz",
        )
        self._regs_npz: Path = PROJECT_ROOT / np_cfg.get(
            "regulations_file", "data/embeddings/regulation_chunks.npz",
        )

        self._job_embeddings: np.ndarray | None = None
        self._job_chunks: list[Chunk] = []
        self._reg_embeddings: np.ndarray | None = None
        self._reg_chunks: list[Chunk] = []

        logger.info("NumpyVectorStore initialized: %s", self._save_dir)

    @property
    def total_chunks(self) -> int:
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
        """Persist job-posting chunks. Drops chunks without an embedding."""
        self._save_collection(chunks, self._jobs_npz, "job")

    def save_regulation_chunks(self, chunks: list[Chunk]) -> None:
        """Persist EU AI Act chunks. Drops chunks without an embedding."""
        self._save_collection(chunks, self._regs_npz, "regulation")

    def _save_collection(
        self, chunks: list[Chunk], npz_path: Path, label: str,
    ) -> None:
        valid = [c for c in chunks if c.embedding is not None]
        skipped = len(chunks) - len(valid)
        if skipped > 0:
            logger.warning(
                "%d %s chunks had no embedding and were excluded",
                skipped, label,
            )
        if not valid:
            logger.warning("No valid %s chunks to save", label)
            return

        matrix = np.asarray([c.embedding for c in valid], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        matrix = matrix / norms

        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(npz_path), embeddings=matrix)

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
        """Load both collections into memory. Missing files = empty."""
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
        meta_path = npz_path.with_name(npz_path.stem + "_meta.json")
        if not npz_path.exists() or not meta_path.exists():
            logger.debug("No %s collection found at %s", label, npz_path)
            return None, []

        data = np.load(str(npz_path))
        matrix = data["embeddings"]

        meta_list = load_json(meta_path)
        chunks: list[Chunk] = []
        for i, d in enumerate(meta_list):
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
        Top-k cosine search across one or both collections.

        Args:
            query_embedding: Query vector (any norm — normalized here).
            source_type: ``"job_posting"`` | ``"eu_ai_act"`` | None.
                None searches both collections.
            top_k: Maximum results returned across all collections.
            filters: Optional metadata filters (see module docstring).

        Returns:
            ``SearchResult`` list sorted by similarity descending.
        """
        q = np.asarray(query_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm

        results: list[SearchResult] = []
        if source_type in (None, "job_posting"):
            results.extend(self._search_collection(
                q, self._job_embeddings, self._job_chunks, filters,
            ))
        if source_type in (None, "eu_ai_act"):
            results.extend(self._search_collection(
                q, self._reg_embeddings, self._reg_chunks, filters,
            ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def _search_collection(
        self,
        query: np.ndarray,
        embeddings: np.ndarray | None,
        chunks: list[Chunk],
        filters: dict[str, Any] | None,
    ) -> list[SearchResult]:
        if embeddings is None or not chunks:
            return []

        if filters:
            indices = self._apply_filters(chunks, filters)
            if not indices:
                return []
            embeddings = embeddings[indices]
            chunks = [chunks[i] for i in indices]

        scores = embeddings @ query
        results: list[SearchResult] = []
        for chunk, score in zip(chunks, scores):
            sim = float(max(0.0, min(1.0, score)))
            results.append(SearchResult(chunk=chunk, similarity=sim))
        return results

    @staticmethod
    def _apply_filters(
        chunks: list[Chunk], filters: dict[str, Any],
    ) -> list[int]:
        """
        Return indices of chunks matching every filter.

        - Scalar metadata + scalar filter: equality.
        - List metadata (skills, domains) + scalar filter: ``in`` check.
        - List metadata + list filter: any-of (set intersection).
        - Scalar metadata + list filter: membership.
        """
        indices: list[int] = []
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata
            match = True
            for key, expected in filters.items():
                actual = getattr(meta, key, None)
                if actual is None:
                    match = False
                    break

                if isinstance(actual, list):
                    if isinstance(expected, list):
                        if not set(actual) & set(expected):
                            match = False
                            break
                    elif expected not in actual:
                        match = False
                        break
                else:
                    if isinstance(expected, list):
                        if actual not in expected:
                            match = False
                            break
                    elif actual != expected:
                        match = False
                        break
            if match:
                indices.append(i)
        return indices
