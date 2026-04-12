"""
Metadata detector processor for Stage 3 enrichment pipeline.

Classifies seniority level, detects German language requirements,
and identifies remote/hybrid work arrangements from job title and
description text.
"""

import re
from typing import Any

from src.models import RawJob
from src.processors.base import BaseProcessor


class MetadataDetector(BaseProcessor):
    """
    Detect seniority, language requirements, and remote work signals.

    Uses keyword-based pattern matching on title and description text.
    Seniority is determined by highest-priority match (lead > senior >
    junior > intern), defaulting to ``mid``.
    """

    # Seniority patterns: checked in priority order (highest first)
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

    # German language REQUIRED patterns (not just preferred/nice-to-have)
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

    # German language PREFERRED patterns (negative — do not flag)
    _GERMAN_PREFERRED = re.compile(
        r"(?:"
        r"german\s+(?:is\s+)?(?:a\s+plus|nice\s+to\s+have|preferred|"
        r"beneficial|advantageous|desirable|helpful|welcome|bonus)"
        r"|(?:nice\s+to\s+have|preferred|bonus)[\s:]+german"
        r")",
        re.IGNORECASE,
    )

    # Remote/hybrid patterns
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
        """
        Detect seniority level from title and description.

        Title matches take priority. Falls back to description matches.
        Default is ``mid`` if no signals are found.

        Args:
            title: Job title.
            text: Cleaned description text.

        Returns:
            One of: intern, junior, mid, senior, lead.
        """
        # Check title first (higher signal)
        for level, pattern in self._SENIORITY_PATTERNS:
            if pattern.search(title):
                return level

        # Check description as fallback
        for level, pattern in self._SENIORITY_PATTERNS:
            if pattern.search(text):
                return level

        return "mid"

    def _detect_german_required(self, text: str) -> bool:
        """
        Detect whether German language is required (not just preferred).

        Args:
            text: Combined title and description text.

        Returns:
            True if German is explicitly required.
        """
        if self._GERMAN_REQUIRED.search(text):
            # Check it's not just "preferred" in the same context
            if not self._GERMAN_PREFERRED.search(text):
                return True
            # If both present, still flag as required (the requirement
            # signal is stronger)
            return True
        return False

    def _detect_remote(self, text: str) -> bool:
        """
        Detect remote or hybrid work arrangement.

        Args:
            text: Combined title and description text.

        Returns:
            True if remote/hybrid signals are found.
        """
        return bool(self._REMOTE.search(text))

    def process(self, job: RawJob, text: str) -> dict[str, Any]:
        """
        Detect seniority, German requirement, and remote status.

        Args:
            job: Raw job object (title used for seniority detection).
            text: Cleaned description text.

        Returns:
            Dict with ``seniority``, ``requires_german``, ``is_remote``.
        """
        search_text = f"{job.title} {text}"

        return {
            "seniority": self._detect_seniority(job.title, text),
            "requires_german": self._detect_german_required(search_text),
            "is_remote": self._detect_remote(search_text),
        }
