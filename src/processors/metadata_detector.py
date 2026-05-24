"""
Metadata detector — classifies seniority, German-language requirement,
and remote/hybrid status from title and description.

Pure regex; no config file required (the patterns rarely change).
Seniority is determined by highest-priority match in this order:
``lead > senior > junior > intern``, defaulting to ``mid`` when no
signal is found. Title matches take precedence over description.

German detection deliberately ignores "nice to have / preferred"
phrasings — only explicit requirements (German "required", "fluent",
"C1/C2", "verhandlungssicher", etc.) flip ``requires_german`` to
True.
"""

import re
from typing import Any

from src.models import RawJob
from src.processors.base import BaseProcessor


class MetadataDetector(BaseProcessor):
    """Detect seniority, German requirement, and remote status."""

    _SENIORITY_PATTERNS: list[tuple[str, re.Pattern]] = [
        ("lead", re.compile(
            r"\b(?:lead|principal|staff|head\s+of|director|vp|"
            r"vice\s+president|chief|distinguished|fellow)\b",
            re.IGNORECASE,
        )),
        ("senior", re.compile(
            r"\b(?:senior|sr\.?|experienced|5\+?\s*years?)\b",
            re.IGNORECASE,
        )),
        ("junior", re.compile(
            r"\b(?:junior|jr\.?|entry[\s-]level|graduate|"
            r"0[\s-]?[12]\s*years?|fresh\s*grad)\b",
            re.IGNORECASE,
        )),
        ("intern", re.compile(
            r"\b(?:intern(?:ship)?|working\s+student|werkstudent|"
            r"trainee|apprentice|praktik\w*)\b",
            re.IGNORECASE,
        )),
    ]

    # German language REQUIRED patterns (not just preferred / nice-to-have)
    _GERMAN_REQUIRED = re.compile(
        r"(?:"
        r"german\s+(?:is\s+)?(?:required|mandatory|must|essential|necessary|a\s+must)"
        r"|(?:required|mandatory|must|fluent)[\s:]+german"
        r"|deutsch\s+(?:erforderlich|zwingend|muss)"
        r"|flie[sß]end(?:es?)?\s+deutsch"
        r"|german\s+(?:fluency|proficiency)\s+required"
        r"|c[12]\s+(?:level\s+)?(?:in\s+)?german"
        r"|verhandlungssicher\w*\s+deutsch"
        r")",
        re.IGNORECASE,
    )

    _REMOTE = re.compile(
        r"\b(?:remote|hybrid|work\s+from\s+home|wfh|"
        r"distributed\s+team|flexible\s+work|telecommut\w+|"
        r"anywhere|home\s+office)\b",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("MetadataDetector initialized")

    def _detect_seniority(self, title: str, text: str) -> str:
        """Title matches win; description is the fallback; default mid."""
        for level, pattern in self._SENIORITY_PATTERNS:
            if pattern.search(title):
                return level
        for level, pattern in self._SENIORITY_PATTERNS:
            if pattern.search(text):
                return level
        return "mid"

    def _detect_german_required(self, text: str) -> bool:
        """True iff the description contains an explicit requirement signal."""
        return bool(self._GERMAN_REQUIRED.search(text))

    def _detect_remote(self, text: str) -> bool:
        """True iff remote/hybrid signals are present."""
        return bool(self._REMOTE.search(text))

    def process(self, job: RawJob, text: str) -> dict[str, Any]:
        """Return ``seniority``, ``requires_german``, ``is_remote``."""
        search_text = f"{job.title} {text}"
        return {
            "seniority": self._detect_seniority(job.title, text),
            "requires_german": self._detect_german_required(search_text),
            "is_remote": self._detect_remote(search_text),
        }
