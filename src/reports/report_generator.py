"""
Markdown report generator.

Renders an ``AnalyticsResult`` into a single markdown document with
four sections (Skills, Architecture-Execution, Governance, Companies)
plus an executive summary at the top. The output is intentionally
plain markdown — no HTML, no images — so it renders cleanly on
GitHub, in PRs, and in static documentation sites.

Usage::

    from src.reports import render_report
    md = render_report(analytics_result)
    Path("data/reports/berlin_ai_talent_radar_report.md").write_text(md)

The CLI ``python main.py report`` calls this end-to-end against the
saved ``data/reports/analytics.json``.
"""

from datetime import datetime
from pathlib import Path

import yaml

from src.models import AnalyticsResult
from src.utils.io import PROJECT_ROOT, save_text
from src.utils.logger import get_logger

logger = get_logger("reports.ReportGenerator")


def render_report(result: AnalyticsResult) -> str:
    """Top-level convenience: build a ReportGenerator and run it."""
    return ReportGenerator().render(result)


class ReportGenerator:
    """Render an AnalyticsResult into markdown."""

    def __init__(self) -> None:
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh)
        self._market_name: str = settings.get("market", {}).get(
            "name", "Berlin AI",
        )
        self._market_location: str = settings.get("market", {}).get(
            "location", "Berlin, Germany",
        )
        self._report_dir: Path = PROJECT_ROOT / settings.get(
            "analytics", {},
        ).get("report_output_dir", "data/reports")
        self._report_file: str = settings.get(
            "analytics", {},
        ).get("report_filename", "berlin_ai_talent_radar_report.md")

    def render(self, result: AnalyticsResult) -> str:
        """Compose the full markdown document."""
        parts = [
            self._header(result),
            self._executive_summary(result),
            self._section_skills(result),
            self._section_arch_exec(result),
            self._section_governance(result),
            self._section_companies(result),
            self._footer(result),
        ]
        return "\n\n".join(p for p in parts if p)

    def save(self, result: AnalyticsResult, output_path: Path | None = None) -> Path:
        """Render and write the report to disk."""
        markdown = self.render(result)
        target = Path(output_path) if output_path else (
            self._report_dir / self._report_file
        )
        save_text(markdown, target)
        return target

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _header(self, result: AnalyticsResult) -> str:
        generated = self._format_timestamp(result.generated_at)
        sources = ", ".join(result.data_sources) or "n/a"
        date_range = result.date_range or {}
        date_line = (
            f"{date_range.get('earliest', '?')} → {date_range.get('latest', '?')}"
            if date_range else "n/a"
        )
        return (
            f"# {self._market_name} Talent Radar — Intelligence Report\n\n"
            f"**Market:** {self._market_location}  \n"
            f"**Generated:** {generated}  \n"
            f"**Postings analyzed:** {result.total_jobs}  \n"
            f"**Data sources:** {sources}  \n"
            f"**Date range:** {date_line}"
        )

    def _executive_summary(self, result: AnalyticsResult) -> str:
        skills = result.skills
        gov = result.governance
        ae = result.arch_exec

        top_skill = (
            f"`{skills.top_20_skills[0][0]}` ({skills.top_20_skills[0][1]} postings)"
            if skills.top_20_skills else "n/a"
        )
        bullets = [
            f"- **{result.total_jobs}** postings analyzed across "
            f"**{result.companies.total_companies}** companies.",
            f"- **{gov.total_ai_roles}** AI roles identified. "
            f"**{gov.high_risk_count}** touch an Annex III high-risk domain.",
            (
                f"- **{gov.governance_gap_count} governance gaps "
                f"({gov.governance_gap_pct}%)** — AI roles in regulated "
                f"domains with zero documented governance awareness. "
                f"**{gov.days_to_enforcement} days** until enforcement "
                f"(2 August 2026)."
            ),
            (
                f"- Architecture-Execution Spectrum mean **{ae.mean_score:.2f}**. "
                f"**{ae.execution_heavy_pct}%** of roles are execution-heavy "
                f"(AI-vulnerable); **{ae.architecture_heavy_pct}%** are "
                f"architecture-heavy (future-proof)."
            ),
            f"- Top in-demand skill: {top_skill}. "
            f"GenAI tooling appears in **{skills.genai_pct}%** of postings.",
        ]
        return "## Executive Summary\n\n" + "\n".join(bullets)

    def _section_skills(self, result: AnalyticsResult) -> str:
        s = result.skills
        if s.total_jobs == 0:
            return "## 1. Skill Landscape\n\n_No data available._"

        lines = [
            "## 1. Skill Landscape",
            "",
            "### Top 20 Skills",
            "",
            "| # | Skill | Postings | % of total |",
            "|---|---|---|---|",
        ]
        for i, (skill, count) in enumerate(s.top_20_skills[:20], start=1):
            pct = s.skill_percentages.get(skill, 0.0)
            lines.append(f"| {i} | `{skill}` | {count} | {pct}% |")

        lines.extend([
            "",
            "### Modality Split",
            "",
            f"- **GenAI tooling**: {s.genai_count} postings ({s.genai_pct}%)",
            f"- **Traditional ML only**: {s.traditional_ml_count} postings "
            f"({s.traditional_ml_pct}%)",
            f"- **Docker / containerization**: {s.docker_count} postings",
            "",
            "### Workplace Signals",
            "",
            f"- **Remote / hybrid**: {s.remote_count} postings ({s.remote_pct}%)",
            f"- **German fluency required**: {s.german_required_count} postings "
            f"({s.german_pct}%)",
            f"- **PhD specified**: {s.phd_count} postings",
            f"- **Master's specified**: {s.masters_count} postings",
        ])

        if s.skill_growth:
            explosive = [k for k, v in s.skill_growth.items() if v == "explosive"]
            declining = [k for k, v in s.skill_growth.items() if v == "declining"]
            if explosive or declining:
                lines.extend(["", "### Growth Signals (HN Who's Hiring, 6-month window)", ""])
                if explosive:
                    lines.append(f"- **Explosive**: {', '.join(f'`{x}`' for x in sorted(explosive))}")
                if declining:
                    lines.append(f"- **Declining**: {', '.join(f'`{x}`' for x in sorted(declining))}")
        return "\n".join(lines)

    def _section_arch_exec(self, result: AnalyticsResult) -> str:
        ae = result.arch_exec
        if ae.total_scored == 0:
            return "## 2. Architecture-Execution Spectrum\n\n_No data available._"

        lines = [
            "## 2. Architecture-Execution Spectrum",
            "",
            "Roles are scored 0.0 (pure execution → AI-vulnerable) to 1.0 "
            "(pure architecture → AI-amplified).",
            "",
            "### Distribution",
            "",
            f"- **Mean score**: {ae.mean_score:.2f} "
            f"(median {ae.median_score:.2f}, σ {ae.std_score:.2f})",
            f"- **Execution-heavy (< 0.40)**: {ae.execution_heavy_count} "
            f"postings ({ae.execution_heavy_pct}%)",
            f"- **Balanced (0.40–0.70)**: {ae.balanced_count} postings",
            f"- **Architecture-heavy (> 0.70)**: {ae.architecture_heavy_count} "
            f"postings ({ae.architecture_heavy_pct}%)",
        ]

        if ae.by_seniority:
            lines.extend(["", "### Mean Score by Seniority", ""])
            order = ["intern", "junior", "mid", "senior", "lead"]
            ordered = sorted(
                ae.by_seniority.items(),
                key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order),
            )
            lines.append("| Level | Mean score |")
            lines.append("|---|---|")
            for level, score in ordered:
                lines.append(f"| {level} | {score:.2f} |")

        if ae.top_architectural_postings:
            lines.extend(["", "### Most Architectural Postings", ""])
            for p in ae.top_architectural_postings[:5]:
                lines.append(
                    f"- **{p['score']:.2f}** — {p['company']}: {p['title']}"
                )
        if ae.top_execution_postings:
            lines.extend(["", "### Most Execution-Heavy Postings", ""])
            for p in ae.top_execution_postings[:5]:
                lines.append(
                    f"- **{p['score']:.2f}** — {p['company']}: {p['title']}"
                )
        return "\n".join(lines)

    def _section_governance(self, result: AnalyticsResult) -> str:
        g = result.governance
        if g.total_ai_roles == 0:
            return "## 3. EU AI Act Governance\n\n_No AI roles in the dataset._"

        lines = [
            "## 3. EU AI Act Governance",
            "",
            f"**Enforcement date:** {g.enforcement_date}  ",
            f"**Days remaining:** {g.days_to_enforcement}  ",
            f"**Maximum penalty:** €{g.max_penalty_eur:,} or 7% of global turnover",
            "",
            "### Headline Numbers",
            "",
            f"- **AI roles:** {g.total_ai_roles}",
            f"- **High-risk roles (Annex III):** {g.high_risk_count} "
            f"({g.high_risk_pct}%)",
            f"- **Mention any governance keyword:** {g.governance_mention_count}",
            f"- **Governance gaps:** {g.governance_gap_count} "
            f"({g.governance_gap_pct}% of high-risk roles)",
        ]

        if g.by_domain:
            lines.extend(["", "### High-Risk Domain Distribution", ""])
            lines.append("| Domain | Postings |")
            lines.append("|---|---|")
            for domain, count in sorted(g.by_domain.items(), key=lambda kv: kv[1], reverse=True):
                lines.append(f"| {domain.replace('_', ' ')} | {count} |")

        if g.article_coverage:
            lines.extend(["", "### Article Coverage", ""])
            lines.append("| Article | Postings mentioning | % of high-risk |")
            lines.append("|---|---|---|")
            for art_num in sorted(g.article_coverage.keys()):
                row = g.article_coverage[art_num]
                lines.append(
                    f"| Article {art_num} | {row.get('postings_mentioning', 0)} "
                    f"| {row.get('pct', 0.0)}% |"
                )

        gap_companies = [
            c for c, data in g.by_company.items()
            if data.get("has_gap")
        ]
        if gap_companies:
            preview = ", ".join(sorted(gap_companies)[:10])
            more = f" (+{len(gap_companies) - 10} more)" if len(gap_companies) > 10 else ""
            lines.extend([
                "",
                "### Companies with at Least One Governance Gap",
                "",
                f"{preview}{more}",
            ])
        return "\n".join(lines)

    def _section_companies(self, result: AnalyticsResult) -> str:
        c = result.companies
        if c.total_companies == 0:
            return "## 4. Company Intelligence\n\n_No data available._"

        lines = [
            "## 4. Company Intelligence",
            "",
            f"**Total distinct companies:** {c.total_companies}",
            "",
            "### Top Employers",
            "",
            "| # | Company | Postings | % | Governance gap | Avg arch-exec |",
            "|---|---|---|---|---|---|",
        ]
        for i, entry in enumerate(c.rankings, start=1):
            company = entry["company"]
            gap_flag = "⚠️ yes" if c.governance_gaps.get(company) else "—"
            avg = c.avg_arch_exec.get(company, 0.0)
            lines.append(
                f"| {i} | {company} | {entry['count']} | "
                f"{entry.get('pct', 0.0)}% | {gap_flag} | {avg:.2f} |"
            )

        if c.skill_profiles:
            lines.extend(["", "### Skill Profiles (Top 8 per company)", ""])
            for company, skills in c.skill_profiles.items():
                if skills:
                    lines.append(
                        f"- **{company}**: {', '.join(f'`{s}`' for s in skills)}"
                    )
        return "\n".join(lines)

    def _footer(self, result: AnalyticsResult) -> str:
        cost = result.cost_summary or {}
        lines = ["---", ""]
        if cost.get("total_eur") is not None:
            lines.append(
                f"_Pipeline cost: €{cost['total_eur']:.4f} "
                f"({cost.get('total_tokens', 0):,} tokens)._"
            )
        lines.append(
            "_Generated by Berlin AI Talent Radar. "
            "EU AI Act references: Regulation EU 2024/1689._"
        )
        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(iso_ts: str) -> str:
        try:
            dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M UTC")
        except (ValueError, TypeError):
            return iso_ts
