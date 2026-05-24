"""
Abstract base class for Stage 3 enrichment processors.

Every per-job processor (cleaner, skill extractor, arch-exec scorer,
governance analyzer, metadata detector) implements this interface.
The pipeline orchestrator calls ``process(job, text)`` for each job
and merges the resulting dicts into a single ``EnrichedJob``.

Design mirrors ``BaseCollector``: small, explicit, testable.

Note on scope: cross-job operations (deduplication, ranking) do NOT
fit this interface — they live in standalone modules such as
``src/processors/deduplicator.py``.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from src.models import RawJob
from src.utils.logger import get_logger


class BaseProcessor(ABC):
    """
    Abstract interface for per-job enrichment processors.

    Subclasses must implement ``process()``. The method returns a dict
    of fields that the pipeline merges into the final ``EnrichedJob``.
    Returning an empty dict is valid (e.g. when no signals match).
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
            job: The raw job object (title, company, etc.).
            text: Cleaned description text. The cleaner runs first and
                  passes its output here for all downstream processors.

        Returns:
            Dict of field names to values, merged into the EnrichedJob.
        """


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
