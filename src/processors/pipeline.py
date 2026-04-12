"""
Enrichment pipeline coordinator for Stage 3.

Chains all five processors (cleaner, skill extractor, arch-exec scorer,
governance analyzer, metadata detector) into a single ``enrich()`` call
that transforms a list of RawJob objects into EnrichedJob objects.

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
from src.processors.governance_analyzer import GovernanceAnalyzer
from src.processors.metadata_detector import MetadataDetector
from src.processors.skill_extractor import SkillExtractor
from src.utils.Io import PROJECT_ROOT, save_json
from src.utils.logger import get_logger

logger = get_logger("processor.EnrichmentPipeline")


class EnrichmentPipeline:
    """
    Orchestrate the full Stage 3 enrichment pipeline.

    Instantiates all processors once and chains them for each job:
    1. Cleaner — strip HTML, normalize, validate length
    2. SkillExtractor — 73-skill regex matching
    3. ArchExecScorer — Architecture-Execution Spectrum
    4. GovernanceAnalyzer — EU AI Act domain + governance gap
    5. MetadataDetector — seniority, German, remote
    """

    def __init__(self) -> None:
        logger.info("Initializing EnrichmentPipeline")
        self._cleaner = Cleaner()
        self._skill_extractor = SkillExtractor()
        self._arch_exec_scorer = ArchExecScorer()
        self._governance_analyzer = GovernanceAnalyzer()
        self._metadata_detector = MetadataDetector()
        logger.info("EnrichmentPipeline ready — all 5 processors loaded")

    def _enrich_one(self, job: RawJob) -> EnrichedJob | None:
        """
        Run all processors on a single RawJob and construct an EnrichedJob.

        Args:
            job: Raw job object from collectors.

        Returns:
            EnrichedJob if successful, None if the job should be skipped.
        """
        # Step 1: Clean text
        clean_result = self._cleaner.process(job, job.description)
        cleaned_text = clean_result["description"]

        # Skip jobs that don't meet minimum length
        if not clean_result["_valid"]:
            logger.debug(
                "Skipping %s — description too short after cleaning", job.source_id
            )
            return None

        # Step 2-5: Run analysis processors on cleaned text
        skills = self._skill_extractor.process(job, cleaned_text)
        arch_exec = self._arch_exec_scorer.process(job, cleaned_text)
        governance = self._governance_analyzer.process(job, cleaned_text)
        metadata = self._metadata_detector.process(job, cleaned_text)

        # Merge all fields into EnrichedJob
        base_fields = job.model_dump()
        base_fields["description"] = cleaned_text  # Use cleaned version

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
        Run the full enrichment pipeline on a batch of raw jobs.

        Args:
            jobs: List of RawJob objects from collectors.
            output_path: Optional path to save enriched jobs as JSON.
                Defaults to ``data/processed/enriched_jobs.json``.

        Returns:
            List of validated EnrichedJob objects.
        """
        logger.info("Starting enrichment pipeline on %d raw jobs", len(jobs))

        enriched: list[EnrichedJob] = []
        skipped = 0
        failed = 0

        for i, job in enumerate(jobs):
            try:
                result = self._enrich_one(job)
                if result is not None:
                    enriched.append(result)
                else:
                    skipped += 1
            except Exception as exc:
                failed += 1
                logger.warning(
                    "Failed to enrich job %s [%s]: %s",
                    job.source_id, job.title, exc,
                )

            # Progress logging every 100 jobs
            if (i + 1) % 100 == 0:
                logger.info("Progress: %d / %d jobs processed", i + 1, len(jobs))

        # Log summary
        self._log_summary(enriched, skipped, failed, len(jobs))

        # Save output
        if output_path is None:
            output_path = PROJECT_ROOT / "data" / "processed" / "enriched_jobs.json"
        save_json([j.model_dump() for j in enriched], output_path)

        return enriched

    def _log_summary(
        self,
        enriched: list[EnrichedJob],
        skipped: int,
        failed: int,
        total: int,
    ) -> None:
        """Log enrichment summary statistics."""
        logger.info("=" * 60)
        logger.info("ENRICHMENT PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info("Total input:    %d", total)
        logger.info("Enriched:       %d", len(enriched))
        logger.info("Skipped (short):%d", skipped)
        logger.info("Failed:         %d", failed)

        if not enriched:
            return

        # Skill stats
        avg_skills = sum(j.skill_count for j in enriched) / len(enriched)
        logger.info("Avg skills/job: %.1f", avg_skills)

        # Arch-exec stats
        avg_score = sum(j.arch_exec_score for j in enriched) / len(enriched)
        logger.info("Avg arch-exec:  %.2f", avg_score)

        # Governance gap stats
        gaps = sum(1 for j in enriched if j.eu_ai_act.governance_gap)
        ai_roles = sum(1 for j in enriched if j.eu_ai_act.is_ai_role)
        logger.info("AI roles:       %d", ai_roles)
        logger.info("Governance gaps: %d", gaps)

        # Seniority distribution
        seniority_counts: dict[str, int] = {}
        for j in enriched:
            seniority_counts[j.seniority] = seniority_counts.get(j.seniority, 0) + 1
        logger.info("Seniority: %s", seniority_counts)
        logger.info("=" * 60)
