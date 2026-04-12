"""
Text cleaner processor for Stage 3 enrichment pipeline.

Strips HTML tags, normalizes whitespace, validates minimum description
length, and truncates to maximum word count.  This runs first in the
pipeline — all downstream processors operate on cleaned text.
"""

import re
from typing import Any

import yaml

from src.models import RawJob
from src.processors.base import BaseProcessor
from src.utils.Io import PROJECT_ROOT


class Cleaner(BaseProcessor):
    """
    Clean and normalize job description text.

    Reads ``min_description_length`` and ``max_description_words`` from
    ``config/settings.yaml`` to control filtering and truncation.
    """

    # Pre-compiled patterns
    _HTML_TAG = re.compile(r"<[^>]+>")
    _HTML_ENTITY = re.compile(r"&\w+;|&#\d+;")
    _MULTI_SPACE = re.compile(r"[ \t]+")
    _MULTI_NEWLINE = re.compile(r"\n{3,}")

    def __init__(self) -> None:
        super().__init__()
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh)
        pipeline = settings.get("pipeline", {})
        self._min_length: int = pipeline.get("min_description_length", 100)
        self._max_words: int = pipeline.get("max_description_words", 5000)
        self.logger.info(
            "Cleaner initialized: min_length=%d, max_words=%d",
            self._min_length, self._max_words,
        )

    def clean(self, text: str) -> str:
        """
        Strip HTML and normalize whitespace.

        Args:
            text: Raw description text (may contain HTML).

        Returns:
            Cleaned plain text.
        """
        # Strip HTML tags
        text = self._HTML_TAG.sub(" ", text)
        # Strip HTML entities
        text = self._HTML_ENTITY.sub(" ", text)
        # Collapse horizontal whitespace
        text = self._MULTI_SPACE.sub(" ", text)
        # Collapse excessive newlines
        text = self._MULTI_NEWLINE.sub("\n\n", text)
        # Strip leading/trailing whitespace per line and overall
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(lines).strip()
        return text

    def truncate(self, text: str) -> str:
        """
        Truncate text to ``max_description_words``.

        Args:
            text: Cleaned text.

        Returns:
            Truncated text if over word limit, otherwise unchanged.
        """
        words = text.split()
        if len(words) > self._max_words:
            self.logger.debug(
                "Truncating description from %d to %d words",
                len(words), self._max_words,
            )
            return " ".join(words[: self._max_words])
        return text

    def is_valid(self, text: str) -> bool:
        """
        Check whether cleaned text meets minimum length requirement.

        Args:
            text: Cleaned text.

        Returns:
            True if text length >= min_description_length.
        """
        return len(text) >= self._min_length

    def process(self, job: RawJob, text: str) -> dict[str, Any]:
        """
        Clean the job description text.

        Args:
            job: Raw job object.
            text: Raw description text.

        Returns:
            Dict with ``description`` key containing cleaned text,
            and ``_valid`` flag indicating whether the job meets
            minimum length requirements.
        """
        cleaned = self.clean(text)
        cleaned = self.truncate(cleaned)
        return {
            "description": cleaned,
            "_valid": self.is_valid(cleaned),
        }
