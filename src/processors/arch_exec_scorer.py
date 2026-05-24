"""
Architecture-Execution Spectrum scorer.

Original analytical framework. Classifies roles on a 0.0 (pure
execution) → 1.0 (pure architecture) scale to predict
AI-vulnerability. Execution-heavy roles (taskwork, tool-specific) are
most exposed to automation; architecture-heavy roles (judgment, system
design) are amplified by AI rather than replaced.

Score formula::

    score = arch_raw / (arch_raw + exec_raw + 0.001)
    clamped to [0.0, 1.0]

Tier 1 signals weight 2.0; tier 2 weight 1.0. The configurable
year-experience regex (e.g. "5+ years experience") is treated as a
tier-1 execution signal when enabled in config.

Config: ``config/arch_exec_signals.yaml``.
"""

import re
from typing import Any

import yaml

from src.models import RawJob
from src.processors.base import BaseProcessor
from src.utils.io import PROJECT_ROOT


class ArchExecScorer(BaseProcessor):
    """Score postings on the Architecture-Execution Spectrum."""

    # Detects "5+ years experience", "3 years of experience", etc.
    _YEAR_EXP_REGEX = re.compile(
        r"\d+\s*\+?\s*years?.{0,30}experience", re.IGNORECASE
    )

    def __init__(self) -> None:
        super().__init__()
        config_path = PROJECT_ROOT / "config" / "arch_exec_signals.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        # [(signal_label, weight, compiled_pattern), ...]
        self._arch_signals: list[tuple[str, float, re.Pattern]] = []
        self._exec_signals: list[tuple[str, float, re.Pattern]] = []

        for tier_key in ("tier1", "tier2"):
            tier = config["architecture_signals"][tier_key]
            weight = float(tier["weight"])
            for signal in tier["signals"]:
                pattern = re.compile(re.escape(signal), re.IGNORECASE)
                self._arch_signals.append((signal, weight, pattern))

        for tier_key in ("tier1", "tier2"):
            tier = config["execution_signals"][tier_key]
            weight = float(tier["weight"])
            for signal in tier["signals"]:
                pattern = re.compile(re.escape(signal), re.IGNORECASE)
                self._exec_signals.append((signal, weight, pattern))

        # Year-experience pattern is treated as execution tier 1 when enabled
        exec_t1 = config["execution_signals"]["tier1"]
        self._use_year_exp: bool = exec_t1.get("use_year_experience_regex", False)
        self._year_exp_weight: float = float(exec_t1["weight"])

        self.logger.info(
            "ArchExecScorer initialized: %d arch signals, %d exec signals",
            len(self._arch_signals), len(self._exec_signals),
        )

    def process(self, job: RawJob, text: str) -> dict[str, Any]:
        """Score one job. Returns normalized score plus raw signals found."""
        search_text = f"{job.title} {text}"

        arch_raw = 0.0
        exec_raw = 0.0
        arch_found: list[str] = []
        exec_found: list[str] = []

        for signal, weight, pattern in self._arch_signals:
            if pattern.search(search_text):
                arch_raw += weight
                arch_found.append(signal)

        for signal, weight, pattern in self._exec_signals:
            if pattern.search(search_text):
                exec_raw += weight
                exec_found.append(signal)

        if self._use_year_exp and self._YEAR_EXP_REGEX.search(search_text):
            exec_raw += self._year_exp_weight
            exec_found.append("year-experience-pattern")

        # Normalize; epsilon keeps the formula safe when both buckets are zero
        score = arch_raw / (arch_raw + exec_raw + 0.001)
        score = max(0.0, min(1.0, score))

        return {
            "arch_exec_score": round(score, 4),
            "arch_signals_found": arch_found,
            "exec_signals_found": exec_found,
            "arch_raw_score": arch_raw,
            "exec_raw_score": exec_raw,
        }
