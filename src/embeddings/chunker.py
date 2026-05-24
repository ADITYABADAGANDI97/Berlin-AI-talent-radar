"""
Word-based overlapping text chunker for Stage 4.

Splits enriched job descriptions and EU AI Act articles into
overlapping chunks with full ``ChunkMetadata`` provenance. Each chunk
carries enough metadata to support filtered vector search and to
reconstruct citations in the RAG layer.

Two flavors:
  - Job postings: chunk_size / overlap / min from ``chunking.job_*``
    in settings.yaml (default 400 / 80 / 50 words).
  - EU AI Act articles: chunk_size / overlap from ``chunking.eu_ai_act_*``
    (default 600 / 50 words). Short articles pass through as a single
    chunk.

Usage::

    from src.embeddings.chunker import Chunker
    chunker = Chunker()
    job_chunks = chunker.chunk_jobs(enriched_jobs)
    reg_chunks = chunker.chunk_eu_articles(articles)
"""

from typing import Any

import yaml

from src.models import Chunk, ChunkMetadata, EnrichedJob
from src.utils.io import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger("embeddings.Chunker")


class Chunker:
    """Word-based overlapping chunker for postings and EU AI Act articles."""

    def __init__(self) -> None:
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh)

        chunking = settings.get("chunking", {})
        self._job_size: int = chunking.get("job_chunk_size_words", 400)
        self._job_overlap: int = chunking.get("job_chunk_overlap_words", 80)
        self._job_min: int = chunking.get("job_chunk_min_words", 50)
        self._eu_size: int = chunking.get("eu_ai_act_chunk_size_words", 600)
        self._eu_overlap: int = chunking.get("eu_ai_act_overlap_words", 50)

        logger.info(
            "Chunker initialized: job=%d/%d/%d, eu=%d/%d",
            self._job_size, self._job_overlap, self._job_min,
            self._eu_size, self._eu_overlap,
        )

    def _split_words(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        min_words: int = 20,
    ) -> list[str]:
        """
        Sliding-window word splitter.

        If the input is at or below ``chunk_size``, returns a single
        chunk (or an empty list when below ``min_words``). For longer
        text, emits overlapping chunks and merges any small tail into
        its predecessor.
        """
        words = text.split()
        if len(words) <= chunk_size:
            return [text] if len(words) >= min_words else []

        step = max(1, chunk_size - overlap)
        chunks: list[str] = []
        for start in range(0, len(words), step):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break

        if len(chunks) > 1 and len(chunks[-1].split()) < min_words:
            merged = chunks[-2] + " " + chunks[-1]
            chunks = chunks[:-2] + [merged]
        return chunks

    @staticmethod
    def _build_job_metadata(job: EnrichedJob) -> ChunkMetadata:
        """Map an EnrichedJob's fields onto ChunkMetadata."""
        return ChunkMetadata(
            source_type="job_posting",
            company=job.company,
            title=job.title,
            location=job.location,
            source=job.source,
            url=job.url,
            skills=job.all_skills_flat or None,
            arch_exec_score=job.arch_exec_score,
            seniority=job.seniority,
            is_high_risk=job.eu_ai_act.touches_high_risk_domain,
            high_risk_domains=job.eu_ai_act.high_risk_domains or None,
            governance_gap=job.eu_ai_act.governance_gap,
            date_posted=job.date_posted,
            hn_month=job.hn_month,
        )

    @staticmethod
    def _build_eu_metadata(article: dict[str, Any]) -> ChunkMetadata:
        """Map an EU AI Act article dict onto ChunkMetadata."""
        return ChunkMetadata(
            source_type="eu_ai_act",
            article_number=article.get("article_number"),
            article_title=article.get("article_title"),
            enforcement_date=article.get("enforcement_date", "2026-08-02"),
            penalty_reference=article.get("penalty_reference"),
        )

    def chunk_job(self, job: EnrichedJob) -> list[Chunk]:
        """Split one EnrichedJob's description into chunks."""
        texts = self._split_words(
            job.description, self._job_size, self._job_overlap, self._job_min,
        )
        if not texts:
            return []

        metadata = self._build_job_metadata(job)
        total = len(texts)
        return [
            Chunk(
                text=t,
                metadata=metadata,
                chunk_index=i,
                total_chunks=total,
            )
            for i, t in enumerate(texts)
        ]

    def chunk_eu_article(self, article: dict[str, Any]) -> list[Chunk]:
        """
        Split one EU AI Act article into chunks.

        Articles shorter than ``min_words`` for the EU bucket pass
        through as a single chunk rather than being dropped — the
        legal text is small enough that losing any article would
        damage retrieval.
        """
        text = article.get("text", "")
        if not text.strip():
            return []

        texts = self._split_words(text, self._eu_size, self._eu_overlap)
        if not texts:
            texts = [text.strip()]

        metadata = self._build_eu_metadata(article)
        total = len(texts)
        return [
            Chunk(
                text=t,
                metadata=metadata,
                chunk_index=i,
                total_chunks=total,
            )
            for i, t in enumerate(texts)
        ]

    def chunk_jobs(self, jobs: list[EnrichedJob]) -> list[Chunk]:
        """Batch-chunk every enriched job."""
        logger.info("Chunking %d enriched jobs", len(jobs))
        all_chunks: list[Chunk] = []
        skipped = 0
        for job in jobs:
            chunks = self.chunk_job(job)
            if chunks:
                all_chunks.extend(chunks)
            else:
                skipped += 1
        logger.info(
            "Job chunking complete: %d chunks from %d jobs (%d skipped)",
            len(all_chunks), len(jobs), skipped,
        )
        return all_chunks

    def chunk_eu_articles(self, articles: list[dict[str, Any]]) -> list[Chunk]:
        """Batch-chunk every EU AI Act article."""
        logger.info("Chunking %d EU AI Act articles", len(articles))
        all_chunks: list[Chunk] = []
        for article in articles:
            all_chunks.extend(self.chunk_eu_article(article))
        logger.info(
            "EU AI Act chunking complete: %d chunks from %d articles",
            len(all_chunks), len(articles),
        )
        return all_chunks
