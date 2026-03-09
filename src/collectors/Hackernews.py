"""
Hacker News "Who is Hiring?" collector for Berlin AI Talent Radar.

Fetches the last 6 monthly "Ask HN: Who is Hiring?" threads via the
HN Algolia search API, then retrieves all top-level comments for each
thread using the official HN Firebase API.

Why HN? It provides 6-month historical data that enables trend analysis
impossible with real-time job boards.  Each posting is tagged with
``hn_month`` (e.g. ``"2025-08"``) so the analytics layer can compute
month-over-month skill frequency changes.

Pipeline:
1. Search Algolia for threads by ``whoishiring`` user.
2. Fetch all top-level comment IDs per thread.
3. Batch-fetch comment text from HN Firebase API.
4. Filter for Berlin/Germany/Remote-EU + AI/data/ML keywords.
5. Parse the free-text "Company | Role | Location | Details" format.
6. Tag each posting with ``hn_month`` for trend analysis.
7. Save to ``data/raw/hackernews.json``.

Expected output: ~150–200 unique postings covering 6 months.
"""

import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import requests

from src.collectors.base import BaseCollector, CollectorError
from src.models import RawJob
from src.utils.io import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ALGOLIA_URL = "https://hn.algolia.com/api/v1/search"
_HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{}.json"
_WHOISHIRING_USER = "whoishiring"
_MAX_MONTHS = 6
_MAX_WORKERS = 10  # Parallel comment fetches
_COMMENT_TIMEOUT = 10  # seconds

# Berlin / Germany / Remote-EU filter keywords
_LOCATION_KEYWORDS = {
    "berlin", "germany", "deutschland", "munich", "münchen",
    "hamburg", "frankfurt", "remote", "eu only", "europe only",
    "anywhere in eu", "emea",
}

# AI / Data / ML filter keywords
_ROLE_KEYWORDS = {
    "machine learning", "ml engineer", "data scientist", "data engineer",
    "ai engineer", "nlp", "llm", "rag", "deep learning", "computer vision",
    "data analyst", "mlops", "generative ai", "genai", "pytorch", "tensorflow",
    "python", "data science", "artificial intelligence", "neural",
    "recommendation", "search engineer", "vector", "embedding",
}


class HackerNewsCollector(BaseCollector):
    """
    Collect Berlin AI job postings from HN "Who is Hiring?" threads.

    Covers the last ``max_months`` monthly threads, enabling 6-month
    skill trend analysis via the ``hn_month`` field on each posting.

    Args:
        config: Full application config dict.
        max_months: Number of past monthly threads to fetch (default: 6).
        output_path: Destination JSON file for raw results.
    """

    def __init__(
        self,
        config: dict[str, Any],
        max_months: int = _MAX_MONTHS,
        output_path: str = "data/raw/hackernews.json",
    ) -> None:
        """
        Initialise the HackerNews collector.

        Args:
            config: Application config dict (settings.yaml content).
            max_months: How many past "Who is Hiring?" threads to fetch.
            output_path: Path to save raw JSON results.
        """
        super().__init__(rate_limit_seconds=0.1)
        self._config = config
        self._max_months = max_months
        self._output_path = output_path

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        """Return source identifier string."""
        return "hackernews"

    def collect(self) -> list[RawJob]:
        """
        Fetch and parse all relevant HN "Who is Hiring?" postings.

        Returns:
            Deduplicated list of ``RawJob`` objects tagged with
            ``hn_month``.

        Example:
            >>> collector = HackerNewsCollector(config=cfg)
            >>> jobs = collector.collect()
            >>> jobs[0].hn_month
            '2025-08'
        """
        threads = self._get_hiring_threads()
        if not threads:
            logger.warning(
                "No 'Who is Hiring?' threads found — HN Algolia may be down."
            )
            return []

        all_jobs: dict[str, RawJob] = {}

        for thread in threads:
            thread_id = thread["objectID"]
            hn_month = thread["hn_month"]
            logger.info(
                "Fetching comments for thread %s (%s)", thread_id, hn_month
            )

            comment_ids = self._get_comment_ids(thread_id)
            logger.info(
                "  Thread %s: %d top-level comments to process",
                thread_id,
                len(comment_ids),
            )

            comments = self._fetch_comments_parallel(comment_ids)
            jobs_in_thread = 0

            for comment in comments:
                text = comment.get("text") or ""
                if not self._is_relevant(text):
                    continue

                job = self._parse_comment(comment, hn_month)
                if job and job.source_id not in all_jobs:
                    all_jobs[job.source_id] = job
                    jobs_in_thread += 1

            logger.info(
                "  Thread %s (%s): %d relevant jobs extracted",
                thread_id,
                hn_month,
                jobs_in_thread,
            )

        result = list(all_jobs.values())
        logger.info(
            "HN collection complete: %d unique jobs across %d threads",
            len(result),
            len(threads),
        )

        save_json([job.model_dump() for job in result], self._output_path)
        return result

    # ------------------------------------------------------------------
    # Private: thread discovery
    # ------------------------------------------------------------------

    def _get_hiring_threads(self) -> list[dict[str, Any]]:
        """
        Find the last ``_max_months`` "Who is Hiring?" threads via Algolia.

        Returns:
            List of dicts with keys ``objectID`` and ``hn_month``
            (``"YYYY-MM"`` string), sorted most-recent first.
        """
        params = {
            "query": "Ask HN: Who is Hiring?",
            "tags": f"story,author_{_WHOISHIRING_USER}",
            "hitsPerPage": 20,
        }
        try:
            response = requests.get(
                _ALGOLIA_URL, params=params, timeout=10
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error("Failed to query HN Algolia API: %s", exc)
            return []

        hits = response.json().get("hits", [])
        threads = []

        for hit in hits:
            title = hit.get("title") or ""
            # Only "Ask HN: Who is Hiring?" — exclude "Who wants to be hired?"
            if "who is hiring" not in title.lower():
                continue

            hn_month = self._extract_month_from_title(title)
            if not hn_month:
                # Fall back to created_at timestamp
                ts = hit.get("created_at_i", 0)
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                hn_month = dt.strftime("%Y-%m")

            threads.append(
                {"objectID": hit["objectID"], "hn_month": hn_month}
            )

            if len(threads) >= self._max_months:
                break

        logger.info("Found %d 'Who is Hiring?' threads", len(threads))
        return threads

    @staticmethod
    def _extract_month_from_title(title: str) -> str | None:
        """
        Extract YYYY-MM from thread titles like "Ask HN: Who is Hiring? (August 2025)".

        Args:
            title: Thread title string.

        Returns:
            ``"YYYY-MM"`` string, or None if not parseable.
        """
        month_map = {
            "january": "01", "february": "02", "march": "03",
            "april": "04", "may": "05", "june": "06",
            "july": "07", "august": "08", "september": "09",
            "october": "10", "november": "11", "december": "12",
        }
        pattern = re.search(
            r"(\b(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\b)\s+(\d{4})",
            title.lower(),
        )
        if not pattern:
            return None
        month_name, year = pattern.group(1), pattern.group(2)
        return f"{year}-{month_map[month_name]}"

    # ------------------------------------------------------------------
    # Private: comment fetching
    # ------------------------------------------------------------------

    def _get_comment_ids(self, thread_id: str) -> list[int]:
        """
        Retrieve all top-level comment IDs for a given thread.

        Args:
            thread_id: HN item ID (string).

        Returns:
            List of integer comment IDs.  Empty list on failure.
        """
        url = _HN_ITEM_URL.format(thread_id)
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("kids", []) or []
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Failed to fetch thread %s: %s", thread_id, exc
            )
            return []

    def _fetch_comments_parallel(
        self, comment_ids: list[int]
    ) -> list[dict[str, Any]]:
        """
        Batch-fetch comment text in parallel using a thread pool.

        Args:
            comment_ids: List of HN item IDs to fetch.

        Returns:
            List of raw comment dicts (``id``, ``text``, ``by``,
            ``time``).  Failed fetches are silently skipped.
        """
        comments: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._fetch_single_comment, cid): cid
                for cid in comment_ids
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    comments.append(result)

        return comments

    def _fetch_single_comment(
        self, comment_id: int
    ) -> dict[str, Any] | None:
        """
        Fetch a single HN comment by its item ID.

        Args:
            comment_id: Integer HN item ID.

        Returns:
            Comment dict, or None if the fetch failed or the comment
            is deleted/dead.
        """
        url = _HN_ITEM_URL.format(comment_id)
        try:
            response = requests.get(url, timeout=_COMMENT_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if data and not data.get("dead") and not data.get("deleted"):
                return data
        except requests.exceptions.RequestException:
            pass
        return None

    # ------------------------------------------------------------------
    # Private: relevance filtering + parsing
    # ------------------------------------------------------------------

    def _is_relevant(self, text: str) -> bool:
        """
        Check if a comment text is a Berlin/EU AI job posting.

        A posting is relevant if it mentions at least one
        location keyword (Berlin/Germany/Remote/EU) AND at least one
        AI/data keyword.

        Args:
            text: Raw HTML-stripped comment text (lowercased internally).

        Returns:
            True if the comment passes both location and role filters.
        """
        if not text or len(text) < 100:
            return False

        text_lower = text.lower()
        has_location = any(kw in text_lower for kw in _LOCATION_KEYWORDS)
        has_role = any(kw in text_lower for kw in _ROLE_KEYWORDS)
        return has_location and has_role

    def _parse_comment(
        self, comment: dict[str, Any], hn_month: str
    ) -> RawJob | None:
        """
        Parse a raw HN comment dict into a ``RawJob`` model.

        HN postings often follow "Company | Role | Location | Details"
        but this is a convention, not a guarantee.  We do best-effort
        extraction and fall back to using the full text as description.

        Args:
            comment: Raw HN comment dict with ``text``, ``by``, ``id``.
            hn_month: ``"YYYY-MM"`` string for this thread.

        Returns:
            Parsed ``RawJob``, or None if the comment lacks a body.
        """
        raw_text = comment.get("text") or ""
        if not raw_text:
            return None

        # Strip HTML tags (HN API returns HTML-encoded text)
        text = re.sub(r"<[^>]+>", " ", raw_text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&quot;", '"', text)
        text = re.sub(r"&#x27;", "'", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Attempt to parse "Company | Role | Location | ..."
        company, title, location = self._extract_header(text)

        comment_id = str(comment.get("id", ""))
        source_id = f"hn_{comment_id}" if comment_id else self._hash_text(text)

        posted_at = None
        ts = comment.get("time")
        if ts:
            try:
                posted_at = datetime.fromtimestamp(
                    int(ts), tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, OSError):
                pass

        url = f"https://news.ycombinator.com/item?id={comment_id}"

        return RawJob(
            company=company,
            title=title,
            location=location,
            description=text,
            date_posted=posted_at,
            url=url,
            source=self.source_name,
            source_id=source_id,
            hn_month=hn_month,
        )

    @staticmethod
    def _extract_header(text: str) -> tuple[str, str, str]:
        """
        Extract company, role, and location from pipe-delimited HN format.

        HN convention: "Company | Role | Location | Remote? | Description"

        Args:
            text: Cleaned comment text.

        Returns:
            Tuple of (company, title, location).  Falls back to
            generic strings if the format is not found.
        """
        # Try pipe-delimited format
        parts = [p.strip() for p in text.split("|")]
        if len(parts) >= 3:
            company = parts[0][:100]
            title = parts[1][:150]
            location = parts[2][:100]
            return company, title, location

        # Try newline-separated header
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) >= 2:
            company = lines[0][:100]
            title = lines[1][:150]
            return company, title, "Berlin / Germany"

        return "Unknown (HN)", text[:100], "Berlin / Germany"

    @staticmethod
    def _hash_text(text: str) -> str:
        """
        Generate a stable source_id from text content.

        Args:
            text: Comment text.

        Returns:
            ``"hn_"`` prefixed 8-char hex digest.
        """
        return "hn_" + hashlib.md5(text[:200].encode()).hexdigest()[:8]