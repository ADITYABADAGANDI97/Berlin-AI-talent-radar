"""
Berlin Startup Jobs collector via the WordPress REST API.

The BSJ site exposes ``/wp-json/wp/v2/posts`` — a clean JSON
endpoint that returns full post bodies, dates, and embedded
taxonomy terms (company, location). This is far more reliable
than HTML scraping, which broke when BSJ refreshed their layout
in early 2026.

Pipeline:
1. Hit ``/wp-json/wp/v2/posts?categories=9`` (Engineering /
   IT / Software Development) — that's where AI/ML/data roles
   live. Pull up to ``max_pages`` × ``per_page`` posts.
2. Use ``_embed=wp:term`` so company and location names come
   back inline — no per-post taxonomy lookups.
3. Filter for AI / ML / data relevance against the title and
   the rendered HTML content (Cleaner downstream will strip
   the HTML before any other processor runs).
4. Map each post to a ``RawJob``.

No auth required. Polite ~1s sleep between page fetches.
"""

import hashlib
from typing import Any

import requests

from src.collectors.base import BaseCollector
from src.models import RawJob
from src.utils.io import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BSJ_REST_URL = "https://berlinstartupjobs.com/wp-json/wp/v2/posts"
# Category id 9 = "IT / Software Development" — that's where the AI/ML/data
# roles live. Discovered from /wp-json/wp/v2/categories.
_ENGINEERING_CATEGORY_ID = 9
_REQUEST_TIMEOUT = 15
_HEADERS = {
    # The site blocks generic / library UAs with 403, so we present
    # as a real browser. No personal info is sent.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# Relevance filter: any of these in title + HTML content qualifies the
# posting as AI / ML / data adjacent. The downstream skill extractor +
# governance analyzer then decides what to do with it.
_ROLE_KEYWORDS = (
    "machine learning", "data scientist", "data engineer", "data analyst",
    "ai engineer", "ml engineer", "mlops", "deep learning", "computer vision",
    "natural language", "nlp", "llm", "rag", "generative ai", "genai",
    "data science", "artificial intelligence", "neural network",
    "pytorch", "tensorflow", "python developer", "data platform",
    "analytics engineer",
)


class BerlinStartupJobsCollector(BaseCollector):
    """
    Pull AI / data / ML postings from Berlin Startup Jobs.

    Uses the WordPress REST API instead of HTML scraping.

    Args:
        config: Full application config dict.
        max_pages: Override the number of REST pages to fetch
            (each page returns up to 50 posts).
        output_path: Destination JSON file for raw results.
    """

    def __init__(
        self,
        config: dict[str, Any],
        max_pages: int | None = None,
        output_path: str = "data/raw/bsj.json",
    ) -> None:
        super().__init__(rate_limit_seconds=1.0)
        self._config = config
        self._output_path = output_path

        bsj_cfg = config.get("collectors", {}).get("berlin_startup_jobs", {})
        self._max_pages: int = max_pages or bsj_cfg.get("max_pages", 3)
        self._per_page: int = 50  # WP REST hard cap is 100; 50 is safe.

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        return "bsj"

    def collect(self) -> list[RawJob]:
        """
        Fetch engineering posts, filter for AI/data relevance, return
        a deduplicated list of ``RawJob``.
        """
        all_jobs: dict[str, RawJob] = {}

        for page in range(1, self._max_pages + 1):
            posts = self._fetch_page(page)
            if posts is None:
                break  # hard failure — stop paginating

            if not posts:
                logger.info("Page %d returned no posts — end of results", page)
                break

            kept = 0
            for raw in posts:
                if not self._is_relevant(raw):
                    continue
                try:
                    job = self._parse_post(raw)
                except Exception as exc:
                    logger.debug(
                        "Skipping post %s: %s", raw.get("id"), exc,
                    )
                    continue
                if job.source_id not in all_jobs:
                    all_jobs[job.source_id] = job
                    kept += 1

            logger.info(
                "  Page %d: %d posts, %d relevant new (total: %d)",
                page, len(posts), kept, len(all_jobs),
            )

            if page < self._max_pages:
                self._sleep()

        result = list(all_jobs.values())
        logger.info("BSJ collection complete: %d unique relevant jobs", len(result))
        save_json([j.model_dump() for j in result], self._output_path)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_page(self, page: int) -> list[dict[str, Any]] | None:
        """
        Fetch one page of posts. Returns the list, ``[]`` for an empty
        page, or ``None`` for a hard failure that should stop pagination.
        """
        params = {
            "categories": _ENGINEERING_CATEGORY_ID,
            "per_page": self._per_page,
            "page": page,
            "_embed": "wp:term",
            "_fields": "id,date,link,title,content,_links,_embedded",
        }
        try:
            response = requests.get(
                _BSJ_REST_URL,
                params=params,
                headers=_HEADERS,
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.exceptions.RequestException as exc:
            logger.warning("BSJ network error on page %d: %s", page, exc)
            return None

        # WP returns 400 when paging past the last page — treat as "done".
        if response.status_code == 400:
            return []

        if response.status_code != 200:
            logger.warning(
                "BSJ HTTP %d on page %d", response.status_code, page,
            )
            return None

        try:
            return response.json()
        except ValueError as exc:
            logger.warning("BSJ response was not JSON: %s", exc)
            return None

    def _is_relevant(self, post: dict[str, Any]) -> bool:
        """True if the post's title or HTML content hits an AI/data keyword."""
        title = (post.get("title") or {}).get("rendered", "").lower()
        content = (post.get("content") or {}).get("rendered", "").lower()
        haystack = f"{title} {content}"
        return any(kw in haystack for kw in _ROLE_KEYWORDS)

    def _parse_post(self, post: dict[str, Any]) -> RawJob:
        """Map a WP REST post payload to a ``RawJob``."""
        title = self._decode_html(
            (post.get("title") or {}).get("rendered", "")
        ).strip() or "Unknown Role"
        content = (post.get("content") or {}).get("rendered", "")

        company, locations = self._extract_terms(post)
        location = ", ".join(locations) if locations else "Berlin, Germany"
        date_posted = self._extract_date(post.get("date"))
        url = post.get("link") or ""

        post_id = post.get("id")
        source_id = (
            f"bsj_{post_id}" if post_id is not None
            else f"bsj_{self._hash_post(title, content)}"
        )

        return RawJob(
            company=company or "Unknown Company",
            title=title,
            location=location,
            description=content or "",
            date_posted=date_posted,
            url=url,
            source=self.source_name,
            source_id=source_id,
        )

    @staticmethod
    def _extract_terms(post: dict[str, Any]) -> tuple[str | None, list[str]]:
        """Pull job_company name and job_location names from _embedded."""
        embedded = post.get("_embedded") or {}
        term_groups = embedded.get("wp:term") or []

        company: str | None = None
        locations: list[str] = []
        for group in term_groups:
            for term in group or []:
                taxonomy = term.get("taxonomy")
                name = term.get("name")
                if not name:
                    continue
                if taxonomy == "job_company" and company is None:
                    company = name
                elif taxonomy == "job_location":
                    locations.append(name)
        return company, locations

    @staticmethod
    def _extract_date(value: Any) -> str | None:
        """WP returns ISO 8601 like '2026-05-07T09:56:25'; keep the date."""
        if not value or not isinstance(value, str):
            return None
        return value.split("T", 1)[0]

    @staticmethod
    def _decode_html(text: str) -> str:
        """Lightweight HTML entity decode for titles."""
        import html
        return html.unescape(text or "")

    @staticmethod
    def _hash_post(title: str, content: str) -> str:
        """Deterministic fallback source_id when the post id is missing."""
        fingerprint = (title + content[:120]).encode("utf-8")
        return hashlib.md5(fingerprint).hexdigest()[:8]
