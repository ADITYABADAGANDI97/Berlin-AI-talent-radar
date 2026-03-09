"""
JSearch API collector for Berlin AI Talent Radar.

JSearch aggregates LinkedIn, Indeed, Glassdoor, and ZipRecruiter into
a single RapidAPI endpoint.  This collector:

1. Reads all search queries from ``config/search_queries.yaml``.
2. Paginate up to ``num_pages`` per query (config-driven).
3. Applies a 1-second inter-request delay to stay within the free-tier
   200 requests/month limit.
4. Deduplicates within the session by ``job_id`` before returning.
5. Saves raw results to ``data/raw/jsearch.json``.

Free tier: 200 requests/month.  At 15 queries × 3 pages = 45 requests,
we have comfortable headroom for daily runs.

Environment variable required: ``JSEARCH_API_KEY`` (RapidAPI key).
"""

import hashlib
import os
import time
from typing import Any

import requests

from src.collectors.base import BaseCollector, CollectorError
from src.models import RawJob
from src.utils.io import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

_JSEARCH_HOST = "jsearch.p.rapidapi.com"
_JSEARCH_BASE_URL = "https://jsearch.p.rapidapi.com/search"


class JSearchCollector(BaseCollector):
    """
    Collect Berlin AI job postings via the JSearch RapidAPI endpoint.

    JSearch aggregates LinkedIn, Indeed, Glassdoor, and ZipRecruiter.
    Results are filtered to Germany/Berlin and the current month.

    Args:
        api_key: RapidAPI key for JSearch.  Falls back to the
            ``JSEARCH_API_KEY`` environment variable if not provided.
        config: Full application config dict loaded from
            ``config/settings.yaml``.
        queries: Optional override list of search query strings.
            If not provided, queries are read from
            ``config/search_queries.yaml`` via the config dict.
        output_path: Where to save the raw JSON results.
    """

    def __init__(
        self,
        config: dict[str, Any],
        api_key: str | None = None,
        queries: list[str] | None = None,
        output_path: str = "data/raw/jsearch.json",
    ) -> None:
        """
        Initialise the JSearch collector.

        Args:
            config: Application config dict (settings.yaml content).
            api_key: RapidAPI key.  Falls back to ``JSEARCH_API_KEY``
                env var.
            queries: Optional search query overrides.
            output_path: Path to save raw JSON results.
        """
        super().__init__(rate_limit_seconds=1.0)
        self._api_key: str = api_key or os.environ.get("JSEARCH_API_KEY", "")
        self._config = config
        self._output_path = output_path

        jsearch_cfg = config.get("collectors", {}).get("jsearch", {})
        self._num_pages: int = jsearch_cfg.get("num_pages", 3)
        self._date_posted: str = jsearch_cfg.get("date_posted", "month")

        # Queries come from config/search_queries.yaml, injected via config
        self._queries: list[str] = queries or config.get("search_queries", [])

        if not self._queries:
            logger.warning(
                "No search queries configured for JSearch. "
                "Check config/search_queries.yaml is loaded into settings."
            )

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        """Return source identifier string."""
        return "jsearch"

    def collect(self) -> list[RawJob]:
        """
        Execute all configured search queries against the JSearch API.

        Iterates over each query in ``self._queries``, paginates to
        ``self._num_pages`` pages, and deduplicates by ``job_id``.

        Returns:
            Deduplicated list of ``RawJob`` objects.

        Raises:
            CollectorError: If no API key is configured.

        Example:
            >>> collector = JSearchCollector(config=cfg)
            >>> jobs = collector.collect()
            >>> len(jobs)  # varies
            423
        """
        if not self._api_key:
            raise CollectorError(
                self.source_name,
                "JSEARCH_API_KEY is not set. Export it as an environment "
                "variable or pass it explicitly.",
            )

        all_jobs: dict[str, RawJob] = {}  # keyed by source_id for dedup
        total_requests = 0

        for query_idx, query in enumerate(self._queries, start=1):
            logger.info(
                "Query %d/%d: '%s'",
                query_idx,
                len(self._queries),
                query,
            )
            for page in range(1, self._num_pages + 1):
                jobs_on_page, requests_made = self._fetch_page(query, page)
                total_requests += requests_made

                new = 0
                for job in jobs_on_page:
                    if job.source_id not in all_jobs:
                        all_jobs[job.source_id] = job
                        new += 1

                logger.info(
                    "  Page %d/%d → %d results, %d new (total unique: %d)",
                    page,
                    self._num_pages,
                    len(jobs_on_page),
                    new,
                    len(all_jobs),
                )

                # If page returned nothing, no point paginating further
                if not jobs_on_page:
                    logger.debug("Empty page — stopping pagination for query")
                    break

                # Rate-limit between pages (skip after last page of last query)
                if not (query_idx == len(self._queries) and page == self._num_pages):
                    self._sleep()

        result = list(all_jobs.values())
        logger.info(
            "JSearch collection complete: %d unique jobs across %d queries "
            "(%d API requests)",
            len(result),
            len(self._queries),
            total_requests,
        )

        save_json([job.model_dump() for job in result], self._output_path)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_page(
        self, query: str, page: int
    ) -> tuple[list[RawJob], int]:
        """
        Fetch a single page of results for one search query.

        Args:
            query: Search query string (e.g. ``"RAG engineer Berlin"``).
            page: 1-based page number.

        Returns:
            Tuple of (list of RawJob, number of HTTP requests made).
            Returns ([], 1) on API errors to allow graceful degradation.
        """
        params = {
            "query": query,
            "page": str(page),
            "num_pages": "1",
            "date_posted": self._date_posted,
            "country": "de",
            "language": "en",
        }
        headers = {
            "X-RapidAPI-Key": self._api_key,
            "X-RapidAPI-Host": _JSEARCH_HOST,
        }

        try:
            response = requests.get(
                _JSEARCH_BASE_URL,
                headers=headers,
                params=params,
                timeout=15,
            )
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Network error fetching JSearch page %d for query '%s': %s",
                page,
                query,
                exc,
            )
            return [], 1

        if response.status_code == 429:
            logger.warning(
                "JSearch rate limit hit (429). Sleeping 60s before retry."
            )
            time.sleep(60)
            try:
                response = requests.get(
                    _JSEARCH_BASE_URL,
                    headers=headers,
                    params=params,
                    timeout=15,
                )
            except requests.exceptions.RequestException as exc:
                logger.error("JSearch retry also failed: %s", exc)
                return [], 1

        if response.status_code != 200:
            logger.warning(
                "JSearch returned HTTP %d for query '%s' page %d",
                response.status_code,
                query,
                page,
            )
            return [], 1

        try:
            payload = response.json()
        except ValueError as exc:
            logger.warning("JSearch response is not valid JSON: %s", exc)
            return [], 1

        raw_jobs = payload.get("data", [])
        jobs = [self._parse_job(j) for j in raw_jobs if self._is_relevant(j)]
        return jobs, 1

    def _is_relevant(self, raw: dict[str, Any]) -> bool:
        """
        Filter jobs to Berlin/Germany that have a usable description.

        Args:
            raw: Raw job dict from JSearch API.

        Returns:
            True if the posting should be included.
        """
        location = (raw.get("job_city") or "").lower()
        country = (raw.get("job_country") or "").lower()
        desc = raw.get("job_description") or ""

        # Must have a meaningful description
        if len(desc.strip()) < 50:
            return False

        # Must be Germany-based (Berlin, Munich, Hamburg, Remote-DE, etc.)
        germany_indicators = {"berlin", "germany", "deutschland", "de"}
        combined = f"{location} {country}"
        return any(ind in combined for ind in germany_indicators)

    def _parse_job(self, raw: dict[str, Any]) -> RawJob:
        """
        Map a raw JSearch API result dict to a ``RawJob`` Pydantic model.

        Args:
            raw: Single job dict from the JSearch ``data`` array.

        Returns:
            Parsed ``RawJob`` instance.
        """
        job_id = raw.get("job_id") or self._fallback_id(raw)
        source_id = f"jsearch_{job_id}"

        # Normalise location
        city = raw.get("job_city") or ""
        state = raw.get("job_state") or ""
        country = raw.get("job_country") or "Germany"
        location_parts = [p for p in [city, state, country] if p]
        location = ", ".join(location_parts) or "Germany"

        return RawJob(
            company=raw.get("employer_name") or "Unknown Company",
            title=raw.get("job_title") or "Unknown Role",
            location=location,
            description=raw.get("job_description") or "",
            date_posted=raw.get("job_posted_at_datetime_utc"),
            url=raw.get("job_apply_link") or raw.get("job_google_link") or "",
            source=self.source_name,
            source_id=source_id,
        )

    @staticmethod
    def _fallback_id(raw: dict[str, Any]) -> str:
        """
        Generate a deterministic fallback ID when ``job_id`` is absent.

        Uses an MD5 hash of company + title + first 100 chars of
        description so the ID is stable across re-runs.

        Args:
            raw: Raw job dict.

        Returns:
            8-character hex string.
        """
        fingerprint = (
            (raw.get("employer_name") or "")
            + (raw.get("job_title") or "")
            + (raw.get("job_description") or "")[:100]
        )
        return hashlib.md5(fingerprint.encode()).hexdigest()[:8]