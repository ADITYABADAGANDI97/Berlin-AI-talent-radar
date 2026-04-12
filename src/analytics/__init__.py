"""
Stage 5: Analytics.

Computes skill demand, governance gaps, arch-exec distributions, and
company intelligence from enriched job data.

Usage::

    from src.analytics import AnalyticsEngine
    engine = AnalyticsEngine()
    result = engine.run(enriched_jobs)
"""

from src.analytics.arch_exec_analytics import compute_arch_exec_analytics
from src.analytics.company_analytics import compute_company_analytics
from src.analytics.engine import AnalyticsEngine
from src.analytics.governance_analytics import compute_governance_analytics
from src.analytics.skill_analytics import compute_skill_analytics

__all__ = [
    "AnalyticsEngine",
    "compute_skill_analytics",
    "compute_governance_analytics",
    "compute_arch_exec_analytics",
    "compute_company_analytics",
]
