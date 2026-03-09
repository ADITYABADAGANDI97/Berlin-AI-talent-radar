"""
Berlin Startup Jobs (BSJ) HTML scraper.

Scrapes https://berlinstartupjobs.com/skill-areas/data-science/ using
BeautifulSoup.  The site lists job cards on paginated listing pages;
each card links to a detail page with the full job description.

Pipeline:
1. Fetch listing pages (up to ``max_pages`` pages, default 3).
2. Extract job card URLs.
3. Follow each URL and scrape the full description from the detail page.
4. Apply a 1-second delay between requests (polite scraping).
5. Normalise into ``RawJob`` models.
6. Save to ``data/raw/bsj.json``.

Expected yield: ~40–60 postings.
No authentication required.
"""

import hashlib
import re
import time
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src.collectors.base import BaseCollector, CollectorError
from src.models import RawJob
from src.utils.io import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BSJ_BASE_URL = "https://berlinstartupjobs.com"
_BSJ_LISTING_URL = "https://berlinstartupjobs.com/skill-areas/data-science/"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; BerlinAITalentRadar/1.0; "
        "+https://github.com/berlin-ai-talent-radar)"
    )
}
_REQUEST_TIMEOUT = 15


class BerlinStartupJobsCollector(BaseCollector):
    """
    Scrape AI/data job postings from Berlin Startup Jobs.

    BSJ is a curated board for Berlin tech startups.  The scraper
    fetches listing cards then follows each detail URL for the full
    description text.

    Args:
        config: Full application config dict.
        max_pages: Number of listing pages to scrape (default: 3).
        output_path: Destination JSON file for raw results.
    """

    def __init__(
        self,
        config: dict[str, Any],
        max_pages: int | None = None,
        output_path: str = "data/raw/bsj.json",
    ) -> None:
        """
        Initialise the BSJ scraper.

        Args:
            config: Application config dict.
            max_pages: Override number of listing pages to scrape.
            output_path: Path to save raw JSON results.
        """
        super().__init__(rate_limit_seconds=1.0)
        self._config = config
        self._output_path = output_path

        bsj_cfg = config.get("collectors", {}).get("berlin_startup_jobs", {})
        self._max_pages: int = max_pages or bsj_cfg.get("max_pages", 3)

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        """Return source identifier string."""
        return "bsj"

    def collect(self) -> list[RawJob]:
        """
        Scrape listing pages, follow detail links, and parse job data.

        Returns:
            Deduplicated list of ``RawJob`` objects.

        Example:
            >>> collector = BerlinStartupJobsCollector(config=cfg)
            >>> jobs = collector.collect()
            >>> len(jobs)
            52
        """
        job_urls = self._collect_listing_urls()
        logger.info(
            "BSJ: found %d job detail URLs across %d listing pages",
            len(job_urls),
            self._max_pages,
        )

        all_jobs: dict[str, RawJob] = {}

        for idx, url in enumerate(job_urls, start=1):
            logger.debug("Scraping job %d/%d: %s", idx, len(job_urls), url)
            job = self._scrape_detail_page(url)
            if job and job.source_id not in all_jobs:
                all_jobs[job.source_id] = job

            # Polite delay between detail page requests
            if idx < len(job_urls):
                self._sleep()

        result = list(all_jobs.values())
        logger.info(
            "BSJ collection complete: %d unique jobs", len(result)
        )

        save_json([job.model_dump() for job in result], self._output_path)
        return result

    # ------------------------------------------------------------------
    # Private: listing page scraping
    # ------------------------------------------------------------------

    def _collect_listing_urls(self) -> list[str]:
        """
        Collect all job detail URLs from BSJ listing pages.

        Returns:
            Deduplicated list of absolute detail page URLs.
        """
        seen_urls: set[str] = set()
        all_urls: list[str] = []

        for page in range(1, self._max_pages + 1):
            page_url = self._listing_page_url(page)
            logger.info("Fetching BSJ listing page %d: %s", page, page_url)

            urls = self._extract_card_urls(page_url)
            new = 0
            for url in urls:
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_urls.append(url)
                    new += 1

            logger.info(
                "  Listing page %d: %d URLs found, %d new", page, len(urls), new
            )

            if not urls:
                logger.info("Empty listing page — stopping pagination")
                break

            if page < self._max_pages:
                self._sleep()

        return all_urls

    def _listing_page_url(self, page: int) -> str:
        """
        Build the paginated listing URL for BSJ data-science category.

        Args:
            page: 1-based page number.

        Returns:
            Absolute URL string.

        Example:
            >>> self._listing_page_url(2)
            'https://berlinstartupjobs.com/skill-areas/data-science/page/2/'
        """
        if page == 1:
            return _BSJ_LISTING_URL
        return f"{_BSJ_BASE_URL}/skill-areas/data-science/page/{page}/"

    def _extract_card_urls(self, listing_url: str) -> list[str]:
        """
        Parse a BSJ listing page HTML and extract job detail URLs.

        BSJ renders job cards as ``<article>`` or ``<li>`` elements
        with an ``<a>`` tag pointing to the detail page.

        Args:
            listing_url: URL of the listing page to scrape.

        Returns:
            List of absolute detail page URLs, or empty list on error.
        """
        html = self._get_html(listing_url)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        urls = []

        # BSJ uses article elements with class containing "job"
        articles = soup.select("article.bsj-job, li.bsj-job, article[id^='post-']")
        if not articles:
            # Fallback: look for any link that points to /job-listings/
            articles = soup.find_all("a", href=re.compile(r"/job-listings?/"))

        if not articles:
            logger.debug(
                "No job cards found on listing page — site structure may have changed"
            )
            return []

        for element in articles:
            # If we got <a> tags directly (fallback path)
            if element.name == "a":
                href = element.get("href", "")
            else:
                # Find the primary link in the card
                link = element.find("a", href=re.compile(r"/job-listings?/"))
                if not link:
                    link = element.find("h2", class_=re.compile(r"title|heading"))
                    if link:
                        link = link.find("a")
                if not link:
                    link = element.find("a")
                if not link:
                    continue
                href = link.get("href", "")

            if not href:
                continue

            absolute = urljoin(_BSJ_BASE_URL, href)
            if "berlinstartupjobs.com" in absolute:
                urls.append(absolute)

        return urls

    # ------------------------------------------------------------------
    # Private: detail page scraping
    # ------------------------------------------------------------------

    def _scrape_detail_page(self, url: str) -> RawJob | None:
        """
        Fetch and parse a single job detail page.

        Args:
            url: Absolute URL of the BSJ job detail page.

        Returns:
            Parsed ``RawJob`` or None if scraping fails.
        """
        html = self._get_html(url)
        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")

        title = self._extract_title(soup)
        company = self._extract_company(soup)
        location = self._extract_location(soup)
        description = self._extract_description(soup)
        date_posted = self._extract_date(soup)

        if not description or len(description) < 100:
            logger.debug("Skipping %s — description too short", url)
            return None

        source_id = f"bsj_{self._hash_url(url)}"

        return RawJob(
            company=company,
            title=title,
            location=location,
            description=description,
            date_posted=date_posted,
            url=url,
            source=self.source_name,
            source_id=source_id,
        )

    # ------------------------------------------------------------------
    # Private: field extractors
    # ------------------------------------------------------------------

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract job title from the detail page."""
        # Try structured job title selectors
        for selector in [
            "h1.job-title",
            "h1.entry-title",
            "h1[class*='title']",
            "h1",
        ]:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)[:200]
        return "Unknown Role"

    def _extract_company(self, soup: BeautifulSoup) -> str:
        """Extract company name from the detail page."""
        for selector in [
            "span.company-name",
            "a[class*='company']",
            "[class*='employer']",
            ".company",
        ]:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)[:150]
        # Fallback: look for structured data
        og_site = soup.find("meta", property="og:site_name")
        if og_site:
            return og_site.get("content", "Unknown Company")[:150]
        return "Unknown Company"

    def _extract_location(self, soup: BeautifulSoup) -> str:
        """Extract job location from the detail page."""
        for selector in [
            "[class*='location']",
            "[class*='city']",
            "span[class*='place']",
        ]:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)[:100]
        return "Berlin, Germany"

    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract the full job description text."""
        # Remove nav, header, footer, sidebar noise first
        for noise in soup.select(
            "nav, header, footer, aside, script, style, [class*='sidebar'], [class*='nav']"
        ):
            noise.decompose()

        for selector in [
            "[class*='job-description']",
            "[class*='description']",
            "[class*='content']",
            "article",
            "main",
        ]:
            element = soup.select_one(selector)
            if element and len(element.get_text(strip=True)) > 200:
                return element.get_text(separator="\n", strip=True)[:8000]
        return ""

    def _extract_date(self, soup: BeautifulSoup) -> str | None:
        """Extract the posting date if available."""
        # Try <time> element
        time_el = soup.find("time")
        if time_el:
            return time_el.get("datetime") or time_el.get_text(strip=True)

        # Try meta
        meta_date = soup.find("meta", property="article:published_time")
        if meta_date:
            return meta_date.get("content")

        return None

    # ------------------------------------------------------------------
    # Private: HTTP helper
    # ------------------------------------------------------------------

    def _get_html(self, url: str) -> str | None:
        """
        Fetch HTML from a URL with error handling.

        Args:
            url: URL to fetch.

        Returns:
            HTML string, or None on failure.
        """
        try:
            response = requests.get(
                url,
                headers=_HEADERS,
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

    @staticmethod
    def _hash_url(url: str) -> str:
        """
        Generate a stable short hash from a URL for use as source_id.

        Args:
            url: Job detail page URL.

        Returns:
            12-character hex string.
        """
        return hashlib.md5(url.encode()).hexdigest()[:12]