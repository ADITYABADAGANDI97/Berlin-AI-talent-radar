"""
Abstract base class for all data collectors.

Every collector in the Berlin AI Talent Radar pipeline implements this
interface.  The contract guarantees:
- A ``collect()`` method that returns a list of ``RawJob`` objects.
- A ``source_name`` property that identifies the data origin.
- Rate-limit awareness via ``_sleep()`` helper.
- Idempotent output: running ``collect()`` twice on the same day
  returns the same logical dataset (deduplication by ``source_id``
  happens downstream in ``cleaner.py``).

Design pattern: Strategy / Template Method.
Swapping a collector means implementing this interface — not touching
any downstream pipeline code.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import final

from src.models import RawJob
from src.utils.logger import get_logger


class BaseCollector(ABC):
    """
    Abstract interface for all job-data collectors.

    Subclasses must implement:
    - ``source_name`` property
    - ``collect()`` method

    They may optionally override ``_sleep()`` to customise rate-limit
    behaviour per source.

    Args:
        rate_limit_seconds: Minimum seconds to sleep between HTTP
            requests.  Override per-source if needed.
    """

    def __init__(self, rate_limit_seconds: float = 1.0) -> None:
        """
        Initialise base collector.

        Args:
            rate_limit_seconds: Delay between successive HTTP requests
                to respect rate limits (default: 1 second).
        """
        self._rate_limit: float = rate_limit_seconds
        self.logger: logging.Logger = get_logger(
            f"collector.{self.__class__.__name__}"
        )

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement these
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def source_name(self) -> str:
        """
        Human-readable identifier for this data source.

        Used as the ``source`` field on every ``RawJob`` produced by
        this collector.  Must be lowercase with no spaces, e.g.
        ``"jsearch"``, ``"hackernews"``, ``"arbeitnow"``.

        Returns:
            String source identifier.
        """

    @abstractmethod
    def collect(self) -> list[RawJob]:
        """
        Fetch raw job postings from the data source.

        Returns:
            List of ``RawJob`` objects.  May be empty if no results are
            found or the API is unavailable.

        Raises:
            CollectorError: For unrecoverable errors (e.g. invalid API
                key).  Transient network errors should be retried
                internally and only raise after exhausting retries.

        Example:
            >>> collector = JSearchCollector(api_key="...", config={})
            >>> jobs = collector.collect()
            >>> len(jobs)
            423
        """

    # ------------------------------------------------------------------
    # Concrete helpers available to all subclasses
    # ------------------------------------------------------------------

    @final
    def _sleep(self, seconds: float | None = None) -> None:
        """
        Sleep for the configured rate-limit delay.

        Subclasses should call this between consecutive HTTP requests
        to respect the data source's rate limits.

        Args:
            seconds: Override the instance-level ``_rate_limit``.  If
                ``None``, the instance default is used.

        Example:
            >>> self._sleep()         # uses default
            >>> self._sleep(2.0)      # override for this request
        """
        delay = seconds if seconds is not None else self._rate_limit
        if delay > 0:
            self.logger.debug("Rate-limit sleep: %.1fs", delay)
            time.sleep(delay)

    @final
    def run(self) -> list[RawJob]:
        """
        Execute the collector with logging bookends.

        Wraps ``collect()`` with structured start/finish log lines that
        include source name and result count.  This is the method
        called by the pipeline orchestrator.

        Returns:
            List of ``RawJob`` objects returned by ``collect()``.

        Example:
            >>> jobs = collector.run()
            2025-08-01T12:00:00 | INFO | collector.JSearchCollector | Starting collection from jsearch
            2025-08-01T12:00:05 | INFO | collector.JSearchCollector | Collected 423 jobs from jsearch
        """
        self.logger.info("Starting collection from %s", self.source_name)
        jobs = self.collect()
        self.logger.info(
            "Collected %d jobs from %s", len(jobs), self.source_name
        )
        return jobs


class CollectorError(Exception):
    """
    Raised when a collector encounters an unrecoverable error.

    Attributes:
        source: Name of the collector that raised the error.
        message: Human-readable description.
        original: Original exception that caused this error (if any).
    """

    def __init__(
        self,
        source: str,
        message: str,
        original: Exception | None = None,
    ) -> None:
        """
        Initialise a CollectorError.

        Args:
            source: Collector source name (e.g. ``"jsearch"``).
            message: Human-readable error description.
            original: Wrapped original exception for chaining.
        """
        self.source = source
        self.original = original
        super().__init__(f"[{source}] {message}")