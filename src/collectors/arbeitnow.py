"""
Arbeitnow public job board API collector.

Arbeitnow exposes a free, no-auth JSON API that aggregates German tech
job postings.  This collector:

1. Paginates the job-board API endpoint.
2. Filters for Berlin-based or remote-DE AI/data/ML roles.
3. Normalises each result into a ``RawJob`` model.
4. Saves raw output to ``data/raw/arbeitnow.json``.

No API key required — Arbeitnow's public endpoint is open.
Expected yield: ~60–80 relevant postings.

API docs: https://www.arbeitnow.com/api/job-board-api
"""

import hashlib
import re
from typing import Any

import requests

from src.collectors.base import BaseCollector, CollectorError
from src.models import RawJob
from src.utils.io import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"
_MAX_PAGES = 15  # Safety cap; real cap set by config

# Keywords used to filter for AI/data/ML relevance
_ROLE_KEYWORDS = {
    "machine learning", "ml ", "data scientist", "data engineer",
    "ai engineer", "nlp ", "llm ", "rag ", "deep learning",
    "computer vision", "data analyst", "mlops", "generative ai",
    "genai", "pytorch", "tensorflow", "python developer",
    "data science", "artificial intelligence", "neural network",
    "recommendation system", "search engineer", "vector database",
    "embedding", "data platform", "analytics engineer",
}

# Location filter: Berlin or remote/Germany
_LOCATION_KEYWORDS = {
    "berlin", "germany", "deutschland", "remote",
    "hybrid", "münchen", "munich",
}


class ArbeitnowCollector(BaseCollector):
    """
    Collect Berlin AI job postings from the Arbeitnow public JSON API.

    Arbeitnow requires no authentication and provides a clean paginated
    API.  Pagination stops when a page returns zero results or we hit
    ``max_pages``.

    Args:
        config: Full application config dict.
        max_pages: Maximum pages to fetch (default: from config or 15).
        output_path: Destination JSON file for raw results.
    """

    def __init__(
        self,
        config: dict[str, Any],
        max_pages: int | None = None,
        output_path: str = "data/raw/arbeitnow.json",
    ) -> None:
        """
        Initialise the Arbeitnow collector.

        Args:
            config: Application config dict (settings.yaml content).
            max_pages: Override maximum pages to paginate.
            output_path: Path to save raw JSON results.
        """
        super().__init__(rate_limit_seconds=0.5)
        self._config = config
        self._output_path = output_path

        arb_cfg = config.get("collectors", {}).get("arbeitnow", {})
        self._max_pages: int = max_pages or arb_cfg.get("max_pages", _MAX_PAGES)

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        """Return source identifier string."""
        return "arbeitnow"

    def collect(self) -> list[RawJob]:
        """
        Paginate the Arbeitnow API and collect relevant jobs.

        Stops pagination when an empty page is returned or ``max_pages``
        is reached.

        Returns:
            Deduplicated list of ``RawJob`` objects.

        Example:
            >>> collector = ArbeitnowCollector(config=cfg)
            >>> jobs = collector.collect()
            >>> len(jobs)
            68
        """
        all_jobs: dict[str, RawJob] = {}

        for page in range(1, self._max_pages + 1):
            logger.info("Fetching Arbeitnow page %d/%d", page, self._max_pages)
            jobs_on_page = self._fetch_page(page)

            if not jobs_on_page:
                logger.info("Empty page at %d — stopping pagination", page)
                break

            new = 0
            for job in jobs_on_page:
                if job.source_id not in all_jobs:
                    all_jobs[job.source_id] = job
                    new += 1

            logger.info(
                "  Page %d: %d results, %d new (total: %d)",
                page,
                len(jobs_on_page),
                new,
                len(all_jobs),
            )

            # If fewer results than expected, we've likely hit the end
            if len(jobs_on_page) < 5:
                logger.info("Sparse page — assuming end of results")
                break

            if page < self._max_pages:
                self._sleep()

        result = list(all_jobs.values())
        logger.info(
            "Arbeitnow collection complete: %d unique relevant jobs", len(result)
        )

        save_json([job.model_dump() for job in result], self._output_path)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_page(self, page: int) -> list[RawJob]:
        """
        Fetch a single page from the Arbeitnow API.

        Args:
            page: 1-based page number.

        Returns:
            List of ``RawJob`` objects that pass relevance filtering.
            Returns empty list on HTTP or parsing errors.
        """
        params = {"page": page}
        try:
            response = requests.get(
                _ARBEITNOW_URL, params=params, timeout=15
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.warning("Arbeitnow request failed (page %d): %s", page, exc)
            return []

        try:
            data = response.json()
        except ValueError as exc:
            logger.warning("Arbeitnow response not valid JSON: %s", exc)
            return []

        raw_jobs = data.get("data", [])
        if not raw_jobs:
            return []

        result = []
        for raw in raw_jobs:
            if self._is_relevant(raw):
                job = self._parse_job(raw)
                result.append(job)

        return result

    def _is_relevant(self, raw: dict[str, Any]) -> bool:
        """
        Filter a raw job dict for Berlin/Germany AI/data/ML relevance.

        Args:
            raw: Raw job dict from Arbeitnow API.

        Returns:
            True if the posting is relevant.
        """
        title = (raw.get("title") or "").lower()
        description = (raw.get("description") or "").lower()
        location = (raw.get("location") or "").lower()
        tags = [t.lower() for t in (raw.get("tags") or [])]
        remote = raw.get("remote", False)

        # Must have a meaningful description
        if len(description.strip()) < 100:
            return False

        # Location filter: Berlin, Germany, or remote
        combined_location = f"{location} {' '.join(tags)}"
        has_location = (
            any(kw in combined_location for kw in _LOCATION_KEYWORDS)
            or remote
        )
        if not has_location:
            return False

        # Role filter: title or description must mention AI/data/ML
        combined_role = f"{title} {description[:1000]}"
        has_role = any(kw in combined_role for kw in _ROLE_KEYWORDS)
        return has_role

    def _parse_job(self, raw: dict[str, Any]) -> RawJob:
        """
        Map a raw Arbeitnow job dict to a ``RawJob`` Pydantic model.

        Args:
            raw: Single job dict from Arbeitnow API response.

        Returns:
            Parsed ``RawJob`` instance.
        """
        slug = raw.get("slug") or self._hash_job(raw)
        source_id = f"arbeitnow_{slug}"

        # Arbeitnow provides a human-readable location string
        location = raw.get("location") or "Germany"
        if raw.get("remote"):
            location = f"{location} (Remote)" if location else "Remote"

        return RawJob(
            company=raw.get("company_name") or "Unknown Company",
            title=raw.get("title") or "Unknown Role",
            location=location,
            description=raw.get("description") or "",
            date_posted=raw.get("created_at") or raw.get("published_at"),
            url=raw.get("url") or "",
            source=self.source_name,
            source_id=source_id,
        )

    @staticmethod
    def _hash_job(raw: dict[str, Any]) -> str:
        """
        Generate a stable fallback source_id from job fields.

        Args:
            raw: Raw job dict.

        Returns:
            8-character hex string.
        """
        fingerprint = (
            (raw.get("company_name") or "")
            + (raw.get("title") or "")
            + (raw.get("description") or "")[:100]
        )
        return hashlib.md5(fingerprint.encode()).hexdigest()[:8]