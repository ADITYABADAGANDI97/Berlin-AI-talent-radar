"""
EU AI Act governance analyzer for Stage 3 enrichment pipeline.

Detects whether a job posting involves an AI role, touches high-risk
domains defined in Annex III, and whether the employer demonstrates
EU AI Act governance awareness. The core output is the ``governance_gap``
flag: True when an AI role operates in a regulated domain with zero
governance keyword mentions.

Config source: ``config/governance_taxonomy.yaml``
"""

import re
from typing import Any

import yaml

from src.models import EUAIActAnalysis, RawJob
from src.processors.base import BaseProcessor
from src.utils.Io import PROJECT_ROOT


class GovernanceAnalyzer(BaseProcessor):
    """
    Analyze EU AI Act governance compliance signals in job postings.

    Loads the governance taxonomy once at init and pre-compiles keyword
    patterns for efficient batch processing.
    """

    def __init__(self) -> None:
        super().__init__()
        config_path = PROJECT_ROOT / "config" / "governance_taxonomy.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        # AI role keywords
        self._ai_keywords: list[re.Pattern] = [
            re.compile(re.escape(kw), re.IGNORECASE)
            for kw in config.get("ai_role_keywords", [])
        ]

        # High-risk domains: {domain_name: {patterns, section, title, articles}}
        self._domains: dict[str, dict] = {}
        for domain_name, spec in config.get("high_risk_domains", {}).items():
            patterns = [
                re.compile(re.escape(kw), re.IGNORECASE)
                for kw in spec["keywords"]
            ]
            self._domains[domain_name] = {
                "patterns": patterns,
                "section": spec["annex_iii_section"],
                "title": spec["annex_iii_title"],
                "articles": spec["articles_triggered"],
            }

        # Governance keywords
        self._gov_keywords: list[tuple[str, re.Pattern]] = [
            (kw, re.compile(re.escape(kw), re.IGNORECASE))
            for kw in config.get("governance_keywords", [])
        ]

        self.logger.info(
            "GovernanceAnalyzer initialized: %d AI keywords, %d domains, %d governance keywords",
            len(self._ai_keywords), len(self._domains), len(self._gov_keywords),
        )

    def process(self, job: RawJob, text: str) -> dict[str, Any]:
        """
        Analyze EU AI Act governance signals in a job posting.

        Args:
            job: Raw job object.
            text: Cleaned description text.

        Returns:
            Dict with ``eu_ai_act`` key containing an EUAIActAnalysis.
        """
        search_text = f"{job.title} {text}"

        # 1. Detect AI role
        is_ai_role = any(p.search(search_text) for p in self._ai_keywords)

        # 2. Detect high-risk domains
        matched_domains: list[str] = []
        matched_sections: list[str] = []
        triggered_articles: set[int] = set()

        for domain_name, spec in self._domains.items():
            for pattern in spec["patterns"]:
                if pattern.search(search_text):
                    matched_domains.append(domain_name)
                    matched_sections.append(
                        f"{spec['section']}: {domain_name}"
                    )
                    triggered_articles.update(spec["articles"])
                    break  # One match per domain is enough

        touches_high_risk = len(matched_domains) > 0

        # 3. Detect governance keywords
        gov_found: list[str] = []
        for kw, pattern in self._gov_keywords:
            if pattern.search(search_text):
                gov_found.append(kw)

        # 4. Compute governance gap
        governance_gap = (
            is_ai_role
            and touches_high_risk
            and len(gov_found) == 0
        )

        analysis = EUAIActAnalysis(
            is_ai_role=is_ai_role,
            touches_high_risk_domain=touches_high_risk,
            high_risk_domains=matched_domains,
            annex_iii_sections=matched_sections,
            governance_keywords_found=gov_found,
            governance_keyword_count=len(gov_found),
            governance_gap=governance_gap,
            relevant_articles=sorted(triggered_articles),
        )

        return {"eu_ai_act": analysis}
