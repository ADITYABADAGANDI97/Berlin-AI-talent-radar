"""
Analytics engine orchestrator.

Runs the four analytics modules (skills, governance, arch-exec,
companies) and bundles their outputs into a single
``AnalyticsResult``. The result is the primary data contract feeding
both the Streamlit dashboard and the markdown report generator.

Usage::

    from src.analytics.engine import AnalyticsEngine
    engine = AnalyticsEngine()
    result = engine.run(enriched_jobs, cost_summary=cost_ledger_summary)
"""

from pathlib import Path
from typing import Any

import yaml

from src.analytics.arch_exec_analytics import compute_arch_exec_analytics
from src.analytics.company_analytics import compute_company_analytics
from src.analytics.governance_analytics import compute_governance_analytics
from src.analytics.skill_analytics import compute_skill_analytics
from src.models import AnalyticsResult, EnrichedJob
from src.utils.io import PROJECT_ROOT, save_json
from src.utils.logger import get_logger

logger = get_logger("analytics.Engine")


class AnalyticsEngine:
    """Orchestrates all analytics computations and persists the result."""

    def __init__(self) -> None:
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh)

        analytics = settings.get("analytics", {})
        self._top_skills: int = analytics.get("top_skills_count", 20)
        self._top_companies: int = analytics.get("top_companies_count", 15)
        self._bins: int = analytics.get("arch_exec_bins", 10)
        self._exec_threshold: float = analytics.get(
            "execution_heavy_threshold", 0.40,
        )
        self._arch_threshold: float = analytics.get(
            "architecture_heavy_threshold", 0.70,
        )
        self._report_dir: Path = PROJECT_ROOT / analytics.get(
            "report_output_dir", "data/reports",
        )
        logger.info("AnalyticsEngine initialized")

    def run(
        self,
        jobs: list[EnrichedJob],
        output_path: str | Path | None = None,
        cost_summary: dict[str, Any] | None = None,
        total_chunks: int = 0,
    ) -> AnalyticsResult:
        """
        Run every analytics module and persist the unified result.

        Args:
            jobs: List of EnrichedJob objects.
            output_path: Optional override; defaults to
                ``data/reports/analytics.json``.
            cost_summary: Optional cost roll-up from the CostLedger.
            total_chunks: Pass-through from the vector store, if loaded.
        """
        logger.info("Running analytics engine on %d jobs", len(jobs))

        skills = compute_skill_analytics(jobs)
        governance = compute_governance_analytics(jobs)
        arch_exec = compute_arch_exec_analytics(
            jobs,
            execution_threshold=self._exec_threshold,
            architecture_threshold=self._arch_threshold,
            top_companies_count=self._top_companies,
            bins=self._bins,
        )
        companies = compute_company_analytics(jobs, top_count=self._top_companies)

        sources = sorted({j.source for j in jobs}) if jobs else []
        dates = [j.date_posted for j in jobs if j.date_posted]
        date_range: dict[str, str] = {}
        if dates:
            ordered = sorted(dates)
            date_range = {"earliest": ordered[0], "latest": ordered[-1]}

        result = AnalyticsResult(
            total_jobs=len(jobs),
            total_chunks=total_chunks,
            data_sources=sources,
            date_range=date_range,
            skills=skills,
            governance=governance,
            arch_exec=arch_exec,
            companies=companies,
            cost_summary=cost_summary or {},
        )

        if output_path is None:
            output_path = self._report_dir / "analytics.json"
        save_json(result.model_dump(), output_path)

        logger.info("Analytics complete. Saved to %s", output_path)
        self._log_summary(result)
        return result

    @staticmethod
    def _log_summary(result: AnalyticsResult) -> None:
        logger.info("=" * 60)
        logger.info("ANALYTICS SUMMARY")
        logger.info("=" * 60)
        logger.info("Total jobs analyzed:  %d", result.total_jobs)
        logger.info("Data sources:         %s", result.data_sources)
        logger.info("Date range:           %s", result.date_range)
        logger.info("---")
        top = result.skills.top_20_skills[0] if result.skills.top_20_skills else None
        logger.info("Top skill:            %s", top if top else "N/A")
        logger.info(
            "GenAI roles:          %d (%.1f%%)",
            result.skills.genai_count, result.skills.genai_pct,
        )
        logger.info("---")
        logger.info("AI roles:             %d", result.governance.total_ai_roles)
        logger.info(
            "Governance gaps:      %d (%.1f%%)",
            result.governance.governance_gap_count,
            result.governance.governance_gap_pct,
        )
        logger.info(
            "Days to enforcement:  %d", result.governance.days_to_enforcement,
        )
        logger.info("---")
        logger.info("Arch-exec mean:       %.2f", result.arch_exec.mean_score)
        logger.info(
            "Execution-heavy:      %d (%.1f%%)",
            result.arch_exec.execution_heavy_count,
            result.arch_exec.execution_heavy_pct,
        )
        logger.info(
            "Architecture-heavy:   %d (%.1f%%)",
            result.arch_exec.architecture_heavy_count,
            result.arch_exec.architecture_heavy_pct,
        )
        logger.info("---")
        logger.info("Total companies:      %d", result.companies.total_companies)
        logger.info("=" * 60)
