"""
EU AI Act governance analytics for Stage 5.

Computes governance gap statistics, high-risk domain distributions,
per-company compliance signals, and article coverage metrics.

Usage::

    from src.analytics.governance_analytics import compute_governance_analytics
    result = compute_governance_analytics(enriched_jobs)
"""

from collections import defaultdict
from datetime import date, datetime
from typing import Any

from src.models import EnrichedJob, GovernanceAnalytics
from src.utils.logger import get_logger

logger = get_logger("analytics.governance")


def compute_governance_analytics(jobs: list[EnrichedJob]) -> GovernanceAnalytics:
    """
    Compute EU AI Act governance analytics from enriched jobs.

    Args:
        jobs: List of EnrichedJob objects with governance analysis complete.

    Returns:
        GovernanceAnalytics with gaps, domain breakdown, and company profiles.
    """
    # Filter to AI roles only
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
        )

    logger.info("Computing governance analytics for %d AI roles", total_ai)

    # High-risk roles
    high_risk_roles = [j for j in ai_roles if j.eu_ai_act.touches_high_risk_domain]
    high_risk_count = len(high_risk_roles)
    high_risk_pct = round(high_risk_count / total_ai * 100, 1)

    # Governance mentions vs gaps
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

    # By domain
    domain_counts: dict[str, int] = defaultdict(int)
    for j in high_risk_roles:
        for domain in j.eu_ai_act.high_risk_domains:
            domain_counts[domain] += 1

    # By company
    company_data: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"ai_roles": 0, "high_risk": 0, "governance_mentions": 0, "has_gap": False}
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

    # Article coverage
    article_counts: dict[int, int] = defaultdict(int)
    for j in high_risk_roles:
        for article in j.eu_ai_act.relevant_articles:
            article_counts[article] += 1

    article_coverage: dict[int, dict[str, Any]] = {}
    for article_num, count in sorted(article_counts.items()):
        article_coverage[article_num] = {
            "postings_mentioning": count,
            "pct": round(count / high_risk_count * 100, 1) if high_risk_count > 0 else 0.0,
        }

    # Days to enforcement
    enforcement = date(2026, 8, 2)
    today = date.today()
    days_remaining = max(0, (enforcement - today).days)

    result = GovernanceAnalytics(
        total_ai_roles=total_ai,
        high_risk_count=high_risk_count,
        high_risk_pct=high_risk_pct,
        governance_mention_count=governance_mention_count,
        governance_gap_count=governance_gap_count,
        governance_gap_pct=governance_gap_pct,
        by_domain=dict(domain_counts),
        by_company=dict(company_data),
        article_coverage=article_coverage,
        enforcement_date="2026-08-02",
        days_to_enforcement=days_remaining,
        max_penalty_eur=35_000_000,
    )

    logger.info(
        "Governance analytics: %d AI roles, %d high-risk, %d gaps (%.1f%%), %d days to enforcement",
        total_ai, high_risk_count, governance_gap_count, governance_gap_pct, days_remaining,
    )
    return result
