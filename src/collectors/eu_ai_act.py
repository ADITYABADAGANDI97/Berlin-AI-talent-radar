"""
EU AI Act text collector / ingestion module.

Rather than scraping a live URL (the official text is stable and well-
defined), this collector loads the EU AI Act article texts from the
structured YAML config file at ``config/eu_ai_act_articles.yaml``.

This approach:
- Is 100% offline — no network required.
- Is idempotent by design — the YAML is version-controlled.
- Gives us full control over exactly which articles to include and how
  they are chunked downstream.

Each article in the YAML becomes one ``Chunk`` object (via the chunker
downstream) with full metadata: article_number, article_title,
enforcement_date, penalty_reference.

The collector does NOT create ``RawJob`` objects — it returns
pre-structured ``Chunk``-compatible dicts so that
``src/processors/chunker.py`` can ingest them directly.

Output: ``data/eu_ai_act/articles.json``
"""

from pathlib import Path
from typing import Any

import yaml

from src.collectors.base import BaseCollector, CollectorError
from src.models import RawJob
from src.utils.io import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_YAML_PATH = "config/eu_ai_act_articles.yaml"
_OUTPUT_PATH = "data/eu_ai_act/articles.json"


class EUAIActCollector(BaseCollector):
    """
    Load EU AI Act article texts from the project's YAML config.

    This is a "collector" in the pipeline sense — it reads source
    material and normalises it for downstream processing — but its
    input is a config file, not a network API.

    The loaded articles are saved as structured JSON dicts (not
    ``RawJob`` objects) to ``data/eu_ai_act/articles.json``.

    Args:
        config: Full application config dict.
        yaml_path: Path to the EU AI Act articles YAML config file.
        output_path: Destination JSON path for serialised articles.
    """

    def __init__(
        self,
        config: dict[str, Any],
        yaml_path: str = _DEFAULT_YAML_PATH,
        output_path: str = _OUTPUT_PATH,
    ) -> None:
        """
        Initialise the EU AI Act collector.

        Args:
            config: Application config dict.
            yaml_path: Path to eu_ai_act_articles.yaml.
            output_path: Path to save parsed article JSON.
        """
        super().__init__(rate_limit_seconds=0.0)  # No network calls
        self._config = config
        self._yaml_path = Path(yaml_path)
        self._output_path = output_path

    # ------------------------------------------------------------------
    # BaseCollector interface
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        """Return source identifier string."""
        return "eu_ai_act"

    def collect(self) -> list[RawJob]:
        """
        Load and validate EU AI Act articles from YAML config.

        This method returns an empty list of ``RawJob`` objects (since
        EU AI Act text is not a job posting) but saves the structured
        article data to disk for the chunker to process.

        The articles are accessed downstream via
        ``load_eu_ai_act_articles()``.

        Returns:
            Empty list (EU AI Act is not job data, but we still
            honour the ``BaseCollector`` interface).

        Raises:
            CollectorError: If the YAML file cannot be loaded or parsed.

        Example:
            >>> collector = EUAIActCollector(config=cfg)
            >>> collector.collect()  # saves articles.json, returns []
            []
        """
        articles = self.load_articles()
        save_json(articles, self._output_path)
        logger.info(
            "EU AI Act: loaded and saved %d articles to %s",
            len(articles),
            self._output_path,
        )
        # Return empty list — EU AI Act text is not a RawJob
        return []

    # ------------------------------------------------------------------
    # Public: article loading
    # ------------------------------------------------------------------

    def load_articles(self) -> list[dict[str, Any]]:
        """
        Load all articles from the YAML config and validate required fields.

        Each article dict in the YAML is expected to have:
        - ``article_number`` (int): EU AI Act article number
        - ``article_title`` (str): Short title of the article
        - ``text`` (str): Full article text
        - ``enforcement_date`` (str): ISO date when this becomes enforceable
        - ``penalty_reference`` (str | None): Article 99 penalty reference

        Args: (none)

        Returns:
            List of validated article dicts.

        Raises:
            CollectorError: If the YAML file is missing or malformed.

        Example:
            >>> articles = collector.load_articles()
            >>> articles[0]["article_number"]
            6
        """
        if not self._yaml_path.exists():
            raise CollectorError(
                self.source_name,
                f"EU AI Act YAML not found: {self._yaml_path}. "
                "Ensure config/eu_ai_act_articles.yaml is present.",
            )

        try:
            with self._yaml_path.open("r", encoding="utf-8") as fh:
                raw_yaml = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise CollectorError(
                self.source_name,
                f"Failed to parse {self._yaml_path}: {exc}",
                original=exc,
            ) from exc

        if not isinstance(raw_yaml, dict):
            raise CollectorError(
                self.source_name,
                f"Expected YAML root to be a dict with an 'articles' key, "
                f"got {type(raw_yaml).__name__}",
            )

        articles_raw = raw_yaml.get("articles", [])
        if not articles_raw:
            logger.warning(
                "No articles found in %s — check YAML structure", self._yaml_path
            )
            return []

        validated = []
        for idx, article in enumerate(articles_raw):
            parsed = self._validate_article(article, idx)
            if parsed:
                validated.append(parsed)

        logger.info(
            "Loaded %d/%d EU AI Act articles from %s",
            len(validated),
            len(articles_raw),
            self._yaml_path,
        )
        return validated

    # ------------------------------------------------------------------
    # Private: validation
    # ------------------------------------------------------------------

    def _validate_article(
        self, article: Any, idx: int
    ) -> dict[str, Any] | None:
        """
        Validate and normalise a single article dict from the YAML.

        Required fields: ``article_number``, ``article_title``, ``text``.
        Optional fields: ``enforcement_date``, ``penalty_reference``,
        ``annex_sections``.

        Args:
            article: Raw article dict from YAML.
            idx: Index in the YAML list (for error messages).

        Returns:
            Normalised article dict, or None if validation fails.
        """
        if not isinstance(article, dict):
            logger.warning("Article at index %d is not a dict — skipping", idx)
            return None

        article_number = article.get("article_number")
        article_title = article.get("article_title") or article.get("title")
        text = article.get("text") or article.get("content")

        if not article_number:
            logger.warning(
                "Article at index %d missing 'article_number' — skipping", idx
            )
            return None

        if not text:
            logger.warning(
                "Article %s missing 'text' — skipping", article_number
            )
            return None

        if not article_title:
            article_title = f"Article {article_number}"
            logger.debug(
                "Article %s missing 'article_title' — using '%s'",
                article_number,
                article_title,
            )

        return {
            "source_type": "eu_ai_act",
            "article_number": int(article_number),
            "article_title": str(article_title),
            "text": str(text).strip(),
            "enforcement_date": article.get("enforcement_date", "2026-08-02"),
            "penalty_reference": article.get("penalty_reference"),
            "annex_sections": article.get("annex_sections", []),
        }


# ---------------------------------------------------------------------------
# Convenience function for pipeline use
# ---------------------------------------------------------------------------

def load_eu_ai_act_articles(
    json_path: str = _OUTPUT_PATH,
    yaml_path: str = _DEFAULT_YAML_PATH,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Load EU AI Act articles from disk (JSON cache) or YAML fallback.

    This function is used by the chunker and RAG engine to access EU AI
    Act text without re-running the full collector.

    Args:
        json_path: Path to the cached JSON output of the collector.
        yaml_path: Path to the source YAML (fallback if JSON missing).
        config: App config dict (only needed for YAML fallback path).

    Returns:
        List of article dicts.

    Example:
        >>> articles = load_eu_ai_act_articles()
        >>> len(articles)
        11
    """
    from src.utils.io import load_json

    json_file = Path(json_path)
    if json_file.exists():
        logger.info("Loading EU AI Act articles from cached JSON: %s", json_path)
        articles = load_json(json_path)
        logger.info("Loaded %d EU AI Act articles", len(articles))
        return articles

    logger.info(
        "EU AI Act JSON cache not found at %s — loading from YAML: %s",
        json_path,
        yaml_path,
    )
    collector = EUAIActCollector(
        config=config or {},
        yaml_path=yaml_path,
        output_path=json_path,
    )
    return collector.load_articles()