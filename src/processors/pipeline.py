"""
Stage 3 enrichment pipeline orchestrator.

Transforms a list of ``RawJob`` into a list of ``EnrichedJob`` by
chaining six steps:

  1. Deduplicate cross-source near-duplicates (list-level).
  2. Clean each description (HTML strip, normalize, truncate).
  3. Skip postings whose cleaned description is too short.
  4. Run analysis processors on cleaned text:
       - SkillExtractor
       - ArchExecScorer
       - GovernanceAnalyzer
       - MetadataDetector
  5. Merge processor outputs into the EnrichedJob constructor.
  6. Persist the result to ``data/processed/enriched_jobs.json``.

Per-job failures are logged and skipped — the pipeline never aborts
mid-batch.

Usage::

    from src.processors.pipeline import EnrichmentPipeline
    pipeline = EnrichmentPipeline()
    enriched = pipeline.enrich(raw_jobs)
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models import EnrichedJob, RawJob
from src.processors.arch_exec_scorer import ArchExecScorer
from src.processors.cleaner import Cleaner
from src.processors.deduplicator import Deduplicator
from src.processors.governance_analyzer import GovernanceAnalyzer
from src.processors.metadata_detector import MetadataDetector
from src.processors.skill_extractor import SkillExtractor
from src.utils.io import PROJECT_ROOT, save_json
from src.utils.logger import get_logger

logger = get_logger("processor.EnrichmentPipeline")


class EnrichmentPipeline:
    """Run the full Stage 3 enrichment pipeline on a batch of raw jobs."""

    def __init__(self) -> None:
        logger.info("Initializing EnrichmentPipeline")
        self._deduplicator = Deduplicator()
        self._cleaner = Cleaner()
        self._skill_extractor = SkillExtractor()
        self._arch_exec_scorer = ArchExecScorer()
        self._governance_analyzer = GovernanceAnalyzer()
        self._metadata_detector = MetadataDetector()
        logger.info("EnrichmentPipeline ready")

    def _enrich_one(self, job: RawJob) -> EnrichedJob | None:
        """Enrich a single RawJob; return None to signal a skip."""
        clean_result = self._cleaner.process(job, job.description)
        cleaned_text = clean_result["description"]

        if not clean_result["_valid"]:
            logger.debug(
                "Skipping %s — description too short after cleaning",
                job.source_id,
            )
            return None

        skills = self._skill_extractor.process(job, cleaned_text)
        arch_exec = self._arch_exec_scorer.process(job, cleaned_text)
        governance = self._governance_analyzer.process(job, cleaned_text)
        metadata = self._metadata_detector.process(job, cleaned_text)

        base_fields = job.model_dump()
        base_fields["description"] = cleaned_text  # cleaned overrides raw

        enriched_fields: dict[str, Any] = {
            **base_fields,
            **skills,
            **arch_exec,
            **governance,
            **metadata,
            "cleaned_at": datetime.now(timezone.utc).isoformat(),
        }
        return EnrichedJob(**enriched_fields)

    def enrich(
        self,
        jobs: list[RawJob],
        output_path: str | Path | None = None,
    ) -> list[EnrichedJob]:
        """
        Run the full enrichment pipeline.

        Args:
            jobs: Raw jobs from collectors.
            output_path: Optional override for the JSON output path.
                Defaults to ``data/processed/enriched_jobs.json``.

        Returns:
            List of validated ``EnrichedJob`` objects.
        """
        logger.info("Starting enrichment on %d raw jobs", len(jobs))

        # Step 1: cross-source fuzzy dedup
        deduped = self._deduplicator.deduplicate(jobs)

        # Steps 2-5: per-job enrichment
        enriched: list[EnrichedJob] = []
        skipped = 0
        failed = 0

        for i, job in enumerate(deduped):
            try:
                result = self._enrich_one(job)
                if result is not None:
                    enriched.append(result)
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                logger.warning(
                    "Failed to enrich %s [%s]: %s",
                    job.source_id, job.title, exc,
                )

            if (i + 1) % 100 == 0:
                logger.info("Progress: %d / %d", i + 1, len(deduped))

        self._log_summary(len(jobs), len(deduped), enriched, skipped, failed)

        # Step 6: persist
        if output_path is None:
            output_path = PROJECT_ROOT / "data" / "processed" / "enriched_jobs.json"
        save_json([j.model_dump() for j in enriched], output_path)

        return enriched

    @staticmethod
    def _log_summary(
        raw_total: int,
        deduped_total: int,
        enriched: list[EnrichedJob],
        skipped: int,
        failed: int,
    ) -> None:
        """Emit an ASCII-box summary of the enrichment run."""
        logger.info("=" * 60)
        logger.info("ENRICHMENT PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info("Raw input:        %d", raw_total)
        logger.info("After dedup:      %d", deduped_total)
        logger.info("Enriched:         %d", len(enriched))
        logger.info("Skipped (short):  %d", skipped)
        logger.info("Failed:           %d", failed)

        if not enriched:
            return

        n = len(enriched)
        avg_skills = sum(j.skill_count for j in enriched) / n
        avg_score = sum(j.arch_exec_score for j in enriched) / n
        ai_roles = sum(1 for j in enriched if j.eu_ai_act.is_ai_role)
        gaps = sum(1 for j in enriched if j.eu_ai_act.governance_gap)

        seniority_counts: dict[str, int] = {}
        for j in enriched:
            seniority_counts[j.seniority] = seniority_counts.get(j.seniority, 0) + 1

        logger.info("Avg skills/job:   %.1f", avg_skills)
        logger.info("Avg arch-exec:    %.2f", avg_score)
        logger.info("AI roles:         %d", ai_roles)
        logger.info("Governance gaps:  %d", gaps)
        logger.info("Seniority:        %s", seniority_counts)
        logger.info("=" * 60)
