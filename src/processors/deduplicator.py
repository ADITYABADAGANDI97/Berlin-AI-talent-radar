"""
Cross-source fuzzy deduplicator.

Within a single source, ``source_id`` uniqueness is enforced by each
collector. This module handles the harder problem: the same posting
syndicated across LinkedIn, Indeed, BSJ, and HN with slightly
different wording, casing, or punctuation in company name and title.

Strategy:
  - Normalize (company, title) to lowercase and collapse whitespace.
  - Bucket candidates by company key for O(n·k) instead of O(n²).
  - Inside each bucket, compare titles pairwise with
    ``difflib.SequenceMatcher.ratio()`` and merge any pair above
    ``dedup_fuzzy_threshold``.
  - When merging, keep the posting with the longer description
    (more downstream signal) and record the dropped ``source_id``
    for traceability.

Config: ``pipeline.dedup_fuzzy_threshold`` from ``config/settings.yaml``
(default 0.85).
"""

import logging
import re
from difflib import SequenceMatcher
from typing import Any

import yaml

from src.models import RawJob
from src.utils.io import PROJECT_ROOT
from src.utils.logger import get_logger


_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def _normalize(value: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    value = value.lower()
    value = _PUNCT.sub(" ", value)
    value = _WHITESPACE.sub(" ", value)
    return value.strip()


def _company_key(company: str) -> str:
    """
    Coarse bucket key for a company name.

    Drops common suffixes (GmbH, Inc, Ltd, ...) so that
    "Acme GmbH" and "Acme" land in the same bucket.
    """
    normalized = _normalize(company)
    for suffix in (" gmbh", " ag", " inc", " ltd", " llc", " bv", " se", " kg"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


class Deduplicator:
    """
    Remove cross-source near-duplicates from a list of ``RawJob``.

    This is a list-level operation, distinct from per-job
    ``BaseProcessor`` instances — hence it does not inherit from
    ``BaseProcessor``.
    """

    def __init__(self, threshold: float | None = None) -> None:
        self.logger: logging.Logger = get_logger(
            f"processor.{self.__class__.__name__}"
        )
        if threshold is None:
            config_path = PROJECT_ROOT / "config" / "settings.yaml"
            with open(config_path, "r", encoding="utf-8") as fh:
                settings = yaml.safe_load(fh)
            threshold = settings.get("pipeline", {}).get(
                "dedup_fuzzy_threshold", 0.85
            )
        self._threshold: float = float(threshold)
        self.logger.info(
            "Deduplicator initialized: fuzzy threshold=%.2f", self._threshold,
        )

    def deduplicate(self, jobs: list[RawJob]) -> list[RawJob]:
        """
        Return a new list with cross-source near-duplicates merged.

        The kept posting per cluster is the one with the longest
        description. Dropped postings are logged at DEBUG level with
        the surviving ``source_id`` for traceability.
        """
        if not jobs:
            return []

        # Bucket by coarse company key
        buckets: dict[str, list[RawJob]] = {}
        for job in jobs:
            buckets.setdefault(_company_key(job.company), []).append(job)

        survivors: list[RawJob] = []
        dropped_count = 0

        for key, bucket in buckets.items():
            kept = self._dedupe_bucket(bucket)
            dropped_count += len(bucket) - len(kept)
            survivors.extend(kept)

        self.logger.info(
            "Deduplication: %d input → %d unique (%d duplicates removed)",
            len(jobs), len(survivors), dropped_count,
        )
        return survivors

    def _dedupe_bucket(self, bucket: list[RawJob]) -> list[RawJob]:
        """Merge near-duplicate titles within a single company bucket."""
        if len(bucket) <= 1:
            return bucket

        # Greedy clustering: each cluster keeps the job with the longest description
        clusters: list[list[RawJob]] = []
        for job in bucket:
            title_norm = _normalize(job.title)
            placed = False
            for cluster in clusters:
                pivot_title = _normalize(cluster[0].title)
                ratio = SequenceMatcher(None, title_norm, pivot_title).ratio()
                if ratio >= self._threshold:
                    cluster.append(job)
                    placed = True
                    break
            if not placed:
                clusters.append([job])

        kept: list[RawJob] = []
        for cluster in clusters:
            if len(cluster) == 1:
                kept.append(cluster[0])
                continue
            # Keep the job with the longest description; log the drops
            winner = max(cluster, key=lambda j: len(j.description))
            losers = [j for j in cluster if j.source_id != winner.source_id]
            for loser in losers:
                self.logger.debug(
                    "Dedup drop: %s [%s] merged into %s [%s]",
                    loser.source_id, loser.source,
                    winner.source_id, winner.source,
                )
            kept.append(winner)
        return kept
