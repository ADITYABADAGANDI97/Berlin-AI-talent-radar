"""
Abstract base class for all Stage 3 processors.

Every processor in the Berlin AI Talent Radar enrichment pipeline
implements this interface.  The contract guarantees:
- A ``process()`` method that returns a dict of enriched fields.
- Structured logging via ``get_logger``.
- Graceful degradation: if a single job fails, skip and continue.

Design pattern: Strategy / Template Method (mirrors BaseCollector).
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, final

from src.models import RawJob
from src.utils.logger import get_logger


class BaseProcessor(ABC):
    """
    Abstract interface for all enrichment processors.

    Subclasses must implement ``process()`` which takes a RawJob and
    cleaned description text, returning a dict of enriched fields to
    merge into the final EnrichedJob.
    """

    def __init__(self) -> None:
        self.logger: logging.Logger = get_logger(
            f"processor.{self.__class__.__name__}"
        )

    @abstractmethod
    def process(self, job: RawJob, text: str) -> dict[str, Any]:
        """
        Extract enrichment fields from a single job posting.

        Args:
            job: The raw job object (for title, company, etc.).
            text: Cleaned description text (HTML stripped, normalized).

        Returns:
            Dict of field names to values, to be merged into EnrichedJob.
        """

    @final
    def run(self, jobs: list[RawJob], texts: list[str]) -> list[dict[str, Any]]:
        """
        Process a batch of jobs with logging and error handling.

        Args:
            jobs: List of RawJob objects.
            texts: Corresponding cleaned description texts.

        Returns:
            List of enrichment dicts (one per job).
        """
        self.logger.info("Starting %s on %d jobs", self.__class__.__name__, len(jobs))
        results: list[dict[str, Any]] = []
        for job, text in zip(jobs, texts):
            try:
                results.append(self.process(job, text))
            except Exception as exc:
                self.logger.warning(
                    "Failed on %s [%s]: %s", job.source_id, job.title, exc
                )
                results.append({})
        self.logger.info("Completed %s on %d jobs", self.__class__.__name__, len(jobs))
        return results


class ProcessorError(Exception):
    """
    Raised when a processor encounters an unrecoverable error.

    Attributes:
        processor: Name of the processor that raised the error.
        message: Human-readable description.
        original: Original exception that caused this error (if any).
    """

    def __init__(
        self,
        processor: str,
        message: str,
        original: Exception | None = None,
    ) -> None:
        self.processor = processor
        self.original = original
        super().__init__(f"[{processor}] {message}")
