"""
LinkedIn job collector via Apify Actor (optional source).

LinkedIn does not provide a public job API.  This collector integrates
with the Apify ``curious_coder/linkedin-jobs-scraper`` actor, which
provides clean structured LinkedIn job data via a REST API.

This collector is OPTIONAL — if no ``APIFY_API_TOKEN`` is configured,
``collect()`` returns an empty list with an INFO log rather than
raising an error.  All other pipeline stages continue normally
(graceful degradation).

Apify free tier: 5 USD/month credit (~2,000 job results).

Environment variable: ``APIFY_API_TOKEN``

Config section (config/settings.yaml):
    collectors:
      linkedin_apify:
        actor_id: "curious_coder/linkedin-jobs-scraper"
        max_results: 200
        location: "Berlin, Germany"
        keywords:
          - "machine learning engineer"
          - "data scientist"
          # ...
"""

import os
import time
from typing import Any

import requests

from src.collectors.base import BaseCollector, CollectorError
from src.models import RawJob
from src.utils.io import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

_APIFY_BASE_URL = "https://api.apify.com/v2"
_DEFAULT_ACTOR_ID = "curious_coder/linkedin-jobs-scraper"
_POLL_INTERVAL_SECONDS = 5
_MAX_POLL_ATTEMPTS = 60  # 5 min max wait


class LinkedInApifyCollector(BaseCollector):
    """
    Collect LinkedIn job postings via the Apify Actor API.

    Apify runs a headless browser actor that scrapes LinkedIn Jobs and
    returns structured JSON.  This collector:

    1. Starts an Apify actor run with our search parameters.
    2. Polls the run status until completion (or timeout).
    3. Fetches the dataset items.
    4. Normalises into ``RawJob`` models.

    If ``APIFY_API_TOKEN`` is not set, the collector returns an empty
    list (graceful degradation — other sources still run).

    Args:
        config: Full application config dict.
        api_token: Apify API token.  Falls back to ``APIFY_API_TOKEN``
            env var.
        output_path: Destination JSON file for raw results.
    """

    def __init__(
        self,
        config: dict[str, Any],
        api_token: str | None = None,
        output_path: str = "data/raw/linkedin.json",
    ) -> None:
        """
        Initialise the LinkedIn / Apify collector.

        Args:
            config: Application config dict.
            api_token: Apify API token (overrides env var).
            output_path: Path to save raw JSON results.
        """
        super().__init__(rate_limit_seconds=0.0)
        self._api_token: str = api_token or os.environ.get("APIFY_API_TOKEN", "")
        self._config = config
        self._output_path = output_path

        li_cfg = config.get("collectors", {}).get("linkedin_apify", {})
        self._actor_id: str = li_cfg.get("actor_id", _DEFAULT_ACTOR_ID)
        self._max_results: int = li_cfg.get("max_results", 200)
        self._location: str = li_cfg.get("location", "Berlin, Germany")
        self._keywords: list[str] = li_cfg.get("keywords", [
            "machine learning engineer",
            "data scientist",
            "AI engineer",
            "data engineer",
            "LLM engineer",
            "MLOps engineer",
        ])

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        """Return source identifier string."""
        return "linkedin"

    def collect(self) -> list[RawJob]:
        """
        Run the Apify actor and collect LinkedIn job postings.

        If ``APIFY_API_TOKEN`` is not configured, returns an empty list
        and logs an INFO message (graceful degradation).

        Returns:
            List of ``RawJob`` objects from LinkedIn.

        Example:
            >>> collector = LinkedInApifyCollector(config=cfg)
            >>> jobs = collector.collect()
        """
        if not self._api_token:
            logger.info(
                "APIFY_API_TOKEN not configured — skipping LinkedIn collection. "
                "Set it to enable LinkedIn as a data source."
            )
            return []

        logger.info(
            "Starting Apify actor '%s' for %d keyword(s) in '%s'",
            self._actor_id,
            len(self._keywords),
            self._location,
        )

        run_id = self._start_actor_run()
        if not run_id:
            logger.warning("Failed to start Apify actor run — skipping LinkedIn")
            return []

        logger.info("Apify run started: %s — polling for completion...", run_id)
        dataset_id = self._wait_for_completion(run_id)
        if not dataset_id:
            logger.warning("Apify run did not complete — skipping LinkedIn results")
            return []

        raw_items = self._fetch_dataset(dataset_id)
        logger.info("Apify returned %d raw items", len(raw_items))

        all_jobs: dict[str, RawJob] = {}
        for item in raw_items:
            job = self._parse_item(item)
            if job and job.source_id not in all_jobs:
                all_jobs[job.source_id] = job

        result = list(all_jobs.values())
        logger.info("LinkedIn collection complete: %d unique jobs", len(result))

        if result:
            save_json([job.model_dump() for job in result], self._output_path)

        return result

    # ------------------------------------------------------------------
    # Private: Apify API
    # ------------------------------------------------------------------

    def _start_actor_run(self) -> str | None:
        """
        Start an Apify actor run with our configured search parameters.

        Returns:
            Run ID string, or None on failure.
        """
        # Build actor input — format follows the curious_coder actor schema
        actor_input = {
            "searchKeywords": self._keywords,
            "location": self._location,
            "maxResults": self._max_results,
            "datePosted": "past-month",
            "jobType": "full-time,contract",
        }

        url = f"{_APIFY_BASE_URL}/acts/{self._actor_id}/runs"
        try:
            response = requests.post(
                url,
                json=actor_input,
                params={"token": self._api_token},
                timeout=30,
            )
            response.raise_for_status()
            run_data = response.json()
            return run_data.get("data", {}).get("id")
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to start Apify actor: %s", exc)
            return None

    def _wait_for_completion(self, run_id: str) -> str | None:
        """
        Poll the Apify run status until it succeeds or times out.

        Args:
            run_id: Apify run ID.

        Returns:
            Dataset ID string if the run succeeded, None otherwise.
        """
        url = f"{_APIFY_BASE_URL}/actor-runs/{run_id}"
        for attempt in range(1, _MAX_POLL_ATTEMPTS + 1):
            try:
                response = requests.get(
                    url,
                    params={"token": self._api_token},
                    timeout=15,
                )
                response.raise_for_status()
                run_data = response.json().get("data", {})
                status = run_data.get("status")

                if status == "SUCCEEDED":
                    dataset_id = run_data.get("defaultDatasetId")
                    logger.info(
                        "Apify run %s succeeded (attempt %d/%d)",
                        run_id,
                        attempt,
                        _MAX_POLL_ATTEMPTS,
                    )
                    return dataset_id

                if status in ("FAILED", "ABORTED", "TIMED-OUT"):
                    logger.warning(
                        "Apify run %s terminated with status: %s", run_id, status
                    )
                    return None

                logger.debug(
                    "Apify run %s status: %s (poll %d/%d)",
                    run_id,
                    status,
                    attempt,
                    _MAX_POLL_ATTEMPTS,
                )
            except requests.exceptions.RequestException as exc:
                logger.warning("Poll attempt %d failed: %s", attempt, exc)

            time.sleep(_POLL_INTERVAL_SECONDS)

        logger.warning(
            "Apify run %s did not complete within %d polls",
            run_id,
            _MAX_POLL_ATTEMPTS,
        )
        return None

    def _fetch_dataset(self, dataset_id: str) -> list[dict[str, Any]]:
        """
        Fetch all items from an Apify dataset.

        Args:
            dataset_id: Apify dataset ID.

        Returns:
            List of raw item dicts, or empty list on failure.
        """
        url = f"{_APIFY_BASE_URL}/datasets/{dataset_id}/items"
        try:
            response = requests.get(
                url,
                params={
                    "token": self._api_token,
                    "format": "json",
                    "limit": self._max_results,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to fetch Apify dataset %s: %s", dataset_id, exc)
            return []

    # ------------------------------------------------------------------
    # Private: normalisation
    # ------------------------------------------------------------------

    def _parse_item(self, item: dict[str, Any]) -> RawJob | None:
        """
        Map a raw Apify LinkedIn item to a ``RawJob`` model.

        Args:
            item: Raw job dict from Apify dataset.

        Returns:
            ``RawJob`` or None if the item lacks essential fields.
        """
        job_id = item.get("jobId") or item.get("id") or ""
        description = item.get("description") or item.get("descriptionText") or ""

        if not description or len(description.strip()) < 100:
            return None

        source_id = f"linkedin_{job_id}" if job_id else f"linkedin_noid_{id(item)}"

        return RawJob(
            company=item.get("companyName") or item.get("company") or "Unknown Company",
            title=item.get("title") or "Unknown Role",
            location=item.get("location") or self._location,
            description=description,
            date_posted=item.get("postedAt") or item.get("publishedAt"),
            url=item.get("jobUrl") or item.get("url") or "",
            source=self.source_name,
            source_id=source_id,
        )