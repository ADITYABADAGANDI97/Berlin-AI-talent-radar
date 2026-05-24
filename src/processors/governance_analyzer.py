"""
EU AI Act governance analyzer.

Detects three signals per posting:
  1. Whether the role is AI/ML/data-science (``is_ai_role``).
  2. Whether it touches an Annex III high-risk domain (mapped to
     section + triggered articles).
  3. Whether the posting demonstrates governance awareness via
     keyword mentions (audit, bias, compliance, etc.).

The headline signal is ``governance_gap``:

    governance_gap = is_ai_role AND touches_high_risk_domain AND
                     governance_keyword_count == 0

A True value means: the company is building/deploying AI in a
regulated domain (Annex III) with zero apparent awareness of their
compliance obligations under Articles 9-15 and 26.

Config: ``config/governance_taxonomy.yaml``.
"""

import re
from typing import Any

import yaml

from src.models import EUAIActAnalysis, RawJob
from src.processors.base import BaseProcessor
from src.utils.io import PROJECT_ROOT


class GovernanceAnalyzer(BaseProcessor):
    """Analyze EU AI Act governance compliance signals."""

    def __init__(self) -> None:
        super().__init__()
        config_path = PROJECT_ROOT / "config" / "governance_taxonomy.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        # AI role keywords — literal substring matching, case-insensitive
        self._ai_keywords: list[re.Pattern] = [
            re.compile(re.escape(kw), re.IGNORECASE)
            for kw in config.get("ai_role_keywords", [])
        ]

        # High-risk domains: {domain_name: {patterns, section, title, articles}}
        self._domains: dict[str, dict[str, Any]] = {}
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

        # Governance keywords with the original surface form preserved
        self._gov_keywords: list[tuple[str, re.Pattern]] = [
            (kw, re.compile(re.escape(kw), re.IGNORECASE))
            for kw in config.get("governance_keywords", [])
        ]

        self.logger.info(
            "GovernanceAnalyzer initialized: %d AI keywords, "
            "%d high-risk domains, %d governance keywords",
            len(self._ai_keywords),
            len(self._domains),
            len(self._gov_keywords),
        )

    def process(self, job: RawJob, text: str) -> dict[str, Any]:
        """Run AI / high-risk / governance detection and return one EUAIActAnalysis."""
        search_text = f"{job.title} {text}"

        # 1. AI role detection
        is_ai_role = any(p.search(search_text) for p in self._ai_keywords)

        # 2. High-risk domain detection
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
                    break  # one match per domain is enough

        touches_high_risk = len(matched_domains) > 0

        # 3. Governance keyword detection
        gov_found: list[str] = [
            kw for kw, pattern in self._gov_keywords if pattern.search(search_text)
        ]

        # 4. Governance gap signal
        governance_gap = (
            is_ai_role and touches_high_risk and len(gov_found) == 0
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
