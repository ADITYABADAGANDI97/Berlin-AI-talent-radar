"""
Analytics engine orchestrator for Stage 5.

Runs all four analytics modules and produces a single AnalyticsResult
for consumption by the Streamlit dashboard and report generator.

Usage::

    from src.analytics.engine import AnalyticsEngine
    engine = AnalyticsEngine()
    result = engine.run(enriched_jobs)
"""

from pathlib import Path

import yaml

from src.analytics.arch_exec_analytics import compute_arch_exec_analytics
from src.analytics.company_analytics import compute_company_analytics
from src.analytics.governance_analytics import compute_governance_analytics
from src.analytics.skill_analytics import compute_skill_analytics
from src.models import AnalyticsResult, EnrichedJob
from src.utils.Io import PROJECT_ROOT, save_json
from src.utils.logger import get_logger

logger = get_logger("analytics.Engine")


class AnalyticsEngine:
    """
    Orchestrate all analytics computations and produce a unified result.

    Loads config for thresholds and output paths, runs all four analytics
    modules, and saves the result as JSON.
    """

    def __init__(self) -> None:
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh)

        analytics = settings.get("analytics", {})
        self._top_skills = analytics.get("top_skills_count", 20)
        self._top_companies = analytics.get("top_companies_count", 15)
        self._bins = analytics.get("arch_exec_bins", 10)
        self._exec_threshold = analytics.get("execution_heavy_threshold", 0.40)
        self._arch_threshold = analytics.get("architecture_heavy_threshold", 0.70)
        self._report_dir = PROJECT_ROOT / analytics.get("report_output_dir", "data/reports")
        self._report_file = analytics.get(
            "report_filename", "berlin_ai_talent_radar_report.md",
        )

        logger.info("AnalyticsEngine initialized")

    def run(
        self,
        jobs: list[EnrichedJob],
        output_path: str | Path | None = None,
        cost_summary: dict | None = None,
    ) -> AnalyticsResult:
        """
        Run all analytics and produce an AnalyticsResult.

        Args:
            jobs: List of EnrichedJob objects.
            output_path: Optional path to save analytics JSON.
                Defaults to data/reports/analytics.json.
            cost_summary: Optional cost summary dict from CostLedger.

        Returns:
            AnalyticsResult with all four analytics modules.
        """
        logger.info("Running analytics engine on %d jobs", len(jobs))

        # Run all four modules
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

        # Compute metadata
        sources = list(set(j.source for j in jobs)) if jobs else []
        dates = [j.date_posted for j in jobs if j.date_posted]
        date_range = {}
        if dates:
            sorted_dates = sorted(dates)
            date_range = {"earliest": sorted_dates[0], "latest": sorted_dates[-1]}

        result = AnalyticsResult(
            total_jobs=len(jobs),
            total_chunks=0,  # Set externally if vector store is available
            data_sources=sources,
            date_range=date_range,
            skills=skills,
            governance=governance,
            arch_exec=arch_exec,
            companies=companies,
            cost_summary=cost_summary or {},
        )

        # Save output
        if output_path is None:
            output_path = self._report_dir / "analytics.json"
        save_json(result.model_dump(), output_path)

        logger.info("Analytics complete. Saved to %s", output_path)
        self._log_summary(result)

        return result

    def _log_summary(self, result: AnalyticsResult) -> None:
        """Log key analytics highlights."""
        logger.info("=" * 60)
        logger.info("ANALYTICS SUMMARY")
        logger.info("=" * 60)
        logger.info("Total jobs analyzed:  %d", result.total_jobs)
        logger.info("Data sources:         %s", result.data_sources)
        logger.info("Date range:           %s", result.date_range)
        logger.info("---")
        logger.info("Top skill:            %s",
                    result.skills.top_20_skills[0] if result.skills.top_20_skills else "N/A")
        logger.info("GenAI roles:          %d (%.1f%%)",
                    result.skills.genai_count, result.skills.genai_pct)
        logger.info("---")
        logger.info("AI roles:             %d", result.governance.total_ai_roles)
        logger.info("Governance gaps:      %d (%.1f%%)",
                    result.governance.governance_gap_count,
                    result.governance.governance_gap_pct)
        logger.info("Days to enforcement:  %d", result.governance.days_to_enforcement)
        logger.info("---")
        logger.info("Arch-exec mean:       %.2f", result.arch_exec.mean_score)
        logger.info("Execution-heavy:      %d (%.1f%%)",
                    result.arch_exec.execution_heavy_count,
                    result.arch_exec.execution_heavy_pct)
        logger.info("Architecture-heavy:   %d (%.1f%%)",
                    result.arch_exec.architecture_heavy_count,
                    result.arch_exec.architecture_heavy_pct)
        logger.info("---")
        logger.info("Total companies:      %d", result.companies.total_companies)
        logger.info("=" * 60)
