"""
Analytics package — Stage 6.

Four pure-function modules (skills, governance, arch-exec, companies)
and one engine that bundles their outputs into an ``AnalyticsResult``
for the dashboard and report generator.
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
