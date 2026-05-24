"""
Text cleaner — first processor in the Stage 3 enrichment pipeline.

Strips HTML tags and entities, normalizes whitespace, truncates to a
configured maximum word count, and validates minimum description
length. All downstream processors operate on the cleaned text this
module produces.

Config: reads ``pipeline.min_description_length`` and
``pipeline.max_description_words`` from ``config/settings.yaml``.
"""

import re
from typing import Any

import yaml

from src.models import RawJob
from src.processors.base import BaseProcessor
from src.utils.io import PROJECT_ROOT


class Cleaner(BaseProcessor):
    """
    Clean and normalize job description text.

    The returned dict carries the cleaned ``description`` plus a
    private ``_valid`` flag the pipeline uses to skip postings whose
    cleaned text is too short to be meaningful.
    """

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
        """Strip HTML and collapse whitespace."""
        text = self._HTML_TAG.sub(" ", text)
        text = self._HTML_ENTITY.sub(" ", text)
        text = self._MULTI_SPACE.sub(" ", text)
        text = self._MULTI_NEWLINE.sub("\n\n", text)
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(lines).strip()

    def truncate(self, text: str) -> str:
        """Truncate to ``max_description_words`` when over the limit."""
        words = text.split()
        if len(words) > self._max_words:
            self.logger.debug(
                "Truncating description from %d to %d words",
                len(words), self._max_words,
            )
            return " ".join(words[: self._max_words])
        return text

    def is_valid(self, text: str) -> bool:
        """True when cleaned text meets the minimum length."""
        return len(text) >= self._min_length

    def process(self, job: RawJob, text: str) -> dict[str, Any]:
        """
        Clean the job description text.

        Returns a dict with ``description`` (cleaned text) and an
        internal ``_valid`` flag the pipeline consumes to drop short
        postings before downstream processors run.
        """
        cleaned = self.clean(text)
        cleaned = self.truncate(cleaned)
        return {
            "description": cleaned,
            "_valid": self.is_valid(cleaned),
        }
