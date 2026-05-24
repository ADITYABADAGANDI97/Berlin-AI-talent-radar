"""
EU AI Act governance analytics.

Pure function: takes a list of ``EnrichedJob`` and returns a
``GovernanceAnalytics`` describing governance-gap prevalence,
high-risk domain distribution, per-company compliance signals,
and article coverage.

The analytics scope is **AI roles only** — non-AI postings can't be
in scope for the EU AI Act and would dilute the rates. Inside that
filter we look at high-risk roles (Annex III) and how many of them
mention any governance keyword.
"""

from collections import defaultdict
from datetime import date
from typing import Any

from src.models import EnrichedJob, GovernanceAnalytics
from src.utils.logger import get_logger

logger = get_logger("analytics.governance")

_ENFORCEMENT_DATE = date(2026, 8, 2)
_MAX_PENALTY_EUR = 35_000_000


def compute_governance_analytics(jobs: list[EnrichedJob]) -> GovernanceAnalytics:
    """Compute EU AI Act governance analytics from enriched jobs."""
    ai_roles = [j for j in jobs if j.eu_ai_act.is_ai_role]
    total_ai = len(ai_roles)

    if total_ai == 0:
        return GovernanceAnalytics(
            total_ai_roles=0,
            high_risk_count=0,
            high_risk_pct=0.0,
            governance_mention_count=0,
            governance_gap_count=0,
            governance_gap_pct=0.0,
            days_to_enforcement=_days_to_enforcement(),
            max_penalty_eur=_MAX_PENALTY_EUR,
        )

    logger.info("Computing governance analytics for %d AI roles", total_ai)

    high_risk_roles = [
        j for j in ai_roles if j.eu_ai_act.touches_high_risk_domain
    ]
    high_risk_count = len(high_risk_roles)
    high_risk_pct = round(high_risk_count / total_ai * 100, 1)

    governance_mention_count = sum(
        1 for j in high_risk_roles if j.eu_ai_act.governance_keyword_count > 0
    )
    governance_gap_count = sum(
        1 for j in high_risk_roles if j.eu_ai_act.governance_gap
    )
    governance_gap_pct = (
        round(governance_gap_count / high_risk_count * 100, 1)
        if high_risk_count > 0 else 0.0
    )

    domain_counts: dict[str, int] = defaultdict(int)
    for j in high_risk_roles:
        for domain in j.eu_ai_act.high_risk_domains:
            domain_counts[domain] += 1

    company_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "ai_roles": 0,
            "high_risk": 0,
            "governance_mentions": 0,
            "has_gap": False,
        }
    )
    for j in ai_roles:
        c = company_data[j.company]
        c["ai_roles"] += 1
        if j.eu_ai_act.touches_high_risk_domain:
            c["high_risk"] += 1
            if j.eu_ai_act.governance_keyword_count > 0:
                c["governance_mentions"] += 1
            if j.eu_ai_act.governance_gap:
                c["has_gap"] = True

    article_counts: dict[int, int] = defaultdict(int)
    for j in high_risk_roles:
        for article in j.eu_ai_act.relevant_articles:
            article_counts[article] += 1

    article_coverage: dict[int, dict[str, Any]] = {}
    for article_num, count in sorted(article_counts.items()):
        article_coverage[article_num] = {
            "postings_mentioning": count,
            "pct": (
                round(count / high_risk_count * 100, 1)
                if high_risk_count > 0 else 0.0
            ),
        }

    days_remaining = _days_to_enforcement()

    result = GovernanceAnalytics(
        total_ai_roles=total_ai,
        high_risk_count=high_risk_count,
        high_risk_pct=high_risk_pct,
        governance_mention_count=governance_mention_count,
        governance_gap_count=governance_gap_count,
        governance_gap_pct=governance_gap_pct,
        by_domain=dict(domain_counts),
        by_company={k: dict(v) for k, v in company_data.items()},
        article_coverage=article_coverage,
        enforcement_date="2026-08-02",
        days_to_enforcement=days_remaining,
        max_penalty_eur=_MAX_PENALTY_EUR,
    )

    logger.info(
        "Governance analytics: %d AI roles, %d high-risk, %d gaps "
        "(%.1f%%), %d days to enforcement",
        total_ai, high_risk_count, governance_gap_count, governance_gap_pct,
        days_remaining,
    )
    return result


def _days_to_enforcement() -> int:
    """Days remaining until the EU AI Act enforcement date."""
    return max(0, (_ENFORCEMENT_DATE - date.today()).days)
