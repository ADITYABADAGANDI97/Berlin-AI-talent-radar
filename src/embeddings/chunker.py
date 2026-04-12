"""
Word-based overlapping text chunker for Stage 4.

Splits enriched job descriptions and EU AI Act articles into
overlapping chunks with full ChunkMetadata provenance. Chunk sizes
and overlaps are config-driven from ``config/settings.yaml``.

Usage::

    from src.embeddings.chunker import Chunker
    chunker = Chunker()
    job_chunks = chunker.chunk_jobs(enriched_jobs)
    reg_chunks = chunker.chunk_eu_articles(articles)
"""

from typing import Any

import yaml

from src.models import Chunk, ChunkMetadata, EnrichedJob
from src.utils.Io import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger("embeddings.Chunker")


class Chunker:
    """
    Word-based overlapping text chunker for job postings and EU AI Act
    articles. Produces Chunk objects with full ChunkMetadata provenance.
    """

    def __init__(self) -> None:
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh)

        chunking = settings.get("chunking", {})
        self._job_size = chunking.get("job_chunk_size_words", 400)
        self._job_overlap = chunking.get("job_chunk_overlap_words", 80)
        self._job_min = chunking.get("job_chunk_min_words", 50)
        self._eu_size = chunking.get("eu_ai_act_chunk_size_words", 600)
        self._eu_overlap = chunking.get("eu_ai_act_overlap_words", 50)

        logger.info(
            "Chunker initialized: job=%d/%d/%d, eu=%d/%d",
            self._job_size, self._job_overlap, self._job_min,
            self._eu_size, self._eu_overlap,
        )

    def _split_words(
        self, text: str, chunk_size: int, overlap: int, min_words: int = 20,
    ) -> list[str]:
        """
        Core sliding-window word splitter.

        Args:
            text: Input text to split.
            chunk_size: Maximum words per chunk.
            overlap: Word overlap between consecutive chunks.
            min_words: Merge final chunk into previous if smaller.

        Returns:
            List of text strings (one per chunk).
        """
        words = text.split()
        if len(words) <= chunk_size:
            return [text] if len(words) >= min_words else []

        step = max(1, chunk_size - overlap)
        chunks: list[str] = []

        for start in range(0, len(words), step):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(chunk_text)
            if end >= len(words):
                break

        # Merge final chunk if too small
        if len(chunks) > 1 and len(chunks[-1].split()) < min_words:
            merged = chunks[-2] + " " + chunks[-1]
            chunks = chunks[:-2] + [merged]

        return chunks

    def _build_job_metadata(self, job: EnrichedJob) -> ChunkMetadata:
        """Build ChunkMetadata from an EnrichedJob."""
        return ChunkMetadata(
            source_type="job_posting",
            company=job.company,
            title=job.title,
            location=job.location,
            source=job.source,
            url=job.url,
            skills=job.all_skills_flat if job.all_skills_flat else None,
            arch_exec_score=job.arch_exec_score,
            seniority=job.seniority,
            is_high_risk=job.eu_ai_act.touches_high_risk_domain,
            high_risk_domains=job.eu_ai_act.high_risk_domains or None,
            governance_gap=job.eu_ai_act.governance_gap,
            date_posted=job.date_posted,
            hn_month=job.hn_month,
        )

    def _build_eu_metadata(self, article: dict[str, Any]) -> ChunkMetadata:
        """Build ChunkMetadata from an EU AI Act article dict."""
        return ChunkMetadata(
            source_type="eu_ai_act",
            article_number=article.get("article_number"),
            article_title=article.get("article_title"),
            enforcement_date=article.get("enforcement_date", "2026-08-02"),
            penalty_reference=article.get("penalty_reference"),
        )

    def chunk_job(self, job: EnrichedJob) -> list[Chunk]:
        """
        Split a single EnrichedJob into overlapping word-based chunks.

        Args:
            job: Enriched job posting with cleaned description.

        Returns:
            List of Chunk objects with source_type="job_posting" metadata.
        """
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
        Split a single EU AI Act article into overlapping chunks.

        Args:
            article: Dict with keys: article_number, article_title, text,
                enforcement_date, penalty_reference.

        Returns:
            List of Chunk objects with source_type="eu_ai_act" metadata.
        """
        text = article.get("text", "")
        if not text.strip():
            return []

        texts = self._split_words(text, self._eu_size, self._eu_overlap)
        if not texts:
            # Article too short for chunking — use as single chunk
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
        """
        Batch-chunk all enriched jobs.

        Args:
            jobs: List of EnrichedJob objects.

        Returns:
            Combined list of all job chunks.
        """
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
        """
        Batch-chunk all EU AI Act articles.

        Args:
            articles: List of article dicts from load_eu_ai_act_articles().

        Returns:
            Combined list of all regulation chunks.
        """
        logger.info("Chunking %d EU AI Act articles", len(articles))
        all_chunks: list[Chunk] = []
        for article in articles:
            chunks = self.chunk_eu_article(article)
            all_chunks.extend(chunks)
        logger.info(
            "EU AI Act chunking complete: %d chunks from %d articles",
            len(all_chunks), len(articles),
        )
        return all_chunks
