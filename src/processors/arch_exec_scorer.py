"""
Architecture-Execution Spectrum scorer for Stage 3 enrichment pipeline.

Applies the original Architecture-Execution Spectrum framework to
classify roles on a 0.0 (pure execution) to 1.0 (pure architecture)
scale. Signals are loaded from ``config/arch_exec_signals.yaml`` with
tier-based weighting (Tier 1 = 2.0x, Tier 2 = 1.0x).

Score formula:
    arch_exec_score = arch_raw / (arch_raw + exec_raw + 0.001)
    Clamped to [0.0, 1.0]
"""

import re
from typing import Any

import yaml

from src.models import RawJob
from src.processors.base import BaseProcessor
from src.utils.Io import PROJECT_ROOT


class ArchExecScorer(BaseProcessor):
    """
    Score jobs on the Architecture-Execution Spectrum.

    Loads signal definitions once at init and pre-compiles regex patterns
    for tier 1 and tier 2 signals in both categories.
    """

    # Year-experience regex from config comment
    _YEAR_EXP_REGEX = re.compile(
        r"\d+\s*\+?\s*years?.{0,30}experience", re.IGNORECASE
    )

    def __init__(self) -> None:
        super().__init__()
        config_path = PROJECT_ROOT / "config" / "arch_exec_signals.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        # Build signal lists: [(pattern_str, weight), ...]
        self._arch_signals: list[tuple[str, float, re.Pattern]] = []
        self._exec_signals: list[tuple[str, float, re.Pattern]] = []

        for tier_key in ("tier1", "tier2"):
            tier = config["architecture_signals"][tier_key]
            weight = tier["weight"]
            for signal in tier["signals"]:
                pattern = re.compile(re.escape(signal), re.IGNORECASE)
                self._arch_signals.append((signal, weight, pattern))

        for tier_key in ("tier1", "tier2"):
            tier = config["execution_signals"][tier_key]
            weight = tier["weight"]
            for signal in tier["signals"]:
                pattern = re.compile(re.escape(signal), re.IGNORECASE)
                self._exec_signals.append((signal, weight, pattern))

        # Check if year-experience regex should be used as exec tier1
        exec_t1 = config["execution_signals"]["tier1"]
        self._use_year_exp = exec_t1.get("use_year_experience_regex", False)
        self._year_exp_weight = exec_t1["weight"]

        self.logger.info(
            "ArchExecScorer initialized: %d arch signals, %d exec signals",
            len(self._arch_signals), len(self._exec_signals),
        )

    def process(self, job: RawJob, text: str) -> dict[str, Any]:
        """
        Score a job on the Architecture-Execution Spectrum.

        Args:
            job: Raw job object.
            text: Cleaned description text.

        Returns:
            Dict with ``arch_exec_score``, ``arch_signals_found``,
            ``exec_signals_found``, ``arch_raw_score``, ``exec_raw_score``.
        """
        search_text = f"{job.title} {text}"

        arch_raw = 0.0
        exec_raw = 0.0
        arch_found: list[str] = []
        exec_found: list[str] = []

        # Match architecture signals
        for signal, weight, pattern in self._arch_signals:
            if pattern.search(search_text):
                arch_raw += weight
                arch_found.append(signal)

        # Match execution signals
        for signal, weight, pattern in self._exec_signals:
            if pattern.search(search_text):
                exec_raw += weight
                exec_found.append(signal)

        # Year-experience regex (exec tier 1)
        if self._use_year_exp and self._YEAR_EXP_REGEX.search(search_text):
            exec_raw += self._year_exp_weight
            exec_found.append("year-experience-pattern")

        # Compute normalized score
        score = arch_raw / (arch_raw + exec_raw + 0.001)
        score = max(0.0, min(1.0, score))

        return {
            "arch_exec_score": round(score, 4),
            "arch_signals_found": arch_found,
            "exec_signals_found": exec_found,
            "arch_raw_score": arch_raw,
            "exec_raw_score": exec_raw,
        }
