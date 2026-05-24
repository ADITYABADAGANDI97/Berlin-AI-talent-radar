"""
Tests for src/analytics/ and src/reports/.

Verify the headline numbers (skill counts, governance-gap rate,
arch-exec distribution, company rankings) on a hand-built dataset,
then render the markdown report and confirm it contains the key
sections and the expected numbers.
"""

from __future__ import annotations

import pytest


def test_skill_analytics_counts_and_split(sample_enriched_dataset):
    from src.analytics import compute_skill_analytics

    s = compute_skill_analytics(sample_enriched_dataset)
    assert s.total_jobs == 12
    # python appears in 10 of the 12 fixture jobs (everyone except n1 and f1)
    assert s.skill_counts.get("python", 0) == 10
    # Deepset's two jobs use RAG/LangChain — GenAI bucket
    assert s.genai_count == 2
    assert s.german_required_count == 1
    assert s.remote_count == 1


def test_governance_analytics_gap_rate(sample_enriched_dataset):
    from src.analytics import compute_governance_analytics

    g = compute_governance_analytics(sample_enriched_dataset)
    # 8 AI roles in the fixture: Acme×3, Deepset×2, Bio Health, TalentBridge, MediScan
    assert g.total_ai_roles == 8
    # High-risk = AI roles touching Annex III domains: Acme×3 + Bio + TalentBridge + MediScan = 6
    assert g.high_risk_count == 6
    # Governance gaps = high-risk with zero gov keywords:
    # Acme×3 (gap=True) + TalentBridge (gap=True) + MediScan (gap=True) = 5
    # Bio Health is high-risk but has gov keywords → not a gap.
    assert g.governance_gap_count == 5
    assert g.governance_gap_pct == round(5 / 6 * 100, 1)
    assert "Essential_Services" in g.by_domain
    assert g.by_company["Acme Bank"]["has_gap"] is True
    assert g.by_company["Bio Health"]["has_gap"] is False
    assert g.days_to_enforcement >= 0


def test_arch_exec_analytics_distribution(sample_enriched_dataset):
    from src.analytics import compute_arch_exec_analytics

    ae = compute_arch_exec_analytics(sample_enriched_dataset)
    assert ae.total_scored == 12
    assert 0.0 <= ae.mean_score <= 1.0
    assert ae.architecture_heavy_count >= 1  # Zalando Principal (0.92)
    assert ae.execution_heavy_count >= 1
    # Senior cohort mean should exceed the intern mean.
    by_sen = ae.by_seniority
    assert by_sen.get("senior", 0) > by_sen.get("intern", 0)


def test_company_analytics_rankings(sample_enriched_dataset):
    from src.analytics import compute_company_analytics

    c = compute_company_analytics(sample_enriched_dataset)
    # Fixture distinct companies: Acme, Deepset, Bio Health, Zalando, N26,
    # Flixbus, TalentBridge, MediScan
    assert c.total_companies == 8
    assert c.rankings[0]["company"] == "Acme Bank"
    assert c.rankings[0]["count"] == 3
    assert c.governance_gaps["Acme Bank"] is True
    assert c.governance_gaps["Bio Health"] is False
    assert "python" in c.skill_profiles["Acme Bank"]


def test_analytics_engine_persists_json(
    tmp_path, sample_enriched_dataset, monkeypatch,
):
    from src.analytics import AnalyticsEngine

    engine = AnalyticsEngine()
    out = tmp_path / "analytics.json"
    result = engine.run(sample_enriched_dataset, output_path=out,
                        cost_summary={"total_eur": 0.01})
    assert out.exists()
    assert result.total_jobs == 12
    assert result.cost_summary == {"total_eur": 0.01}


def test_report_generator_produces_markdown(sample_enriched_dataset):
    from src.analytics import AnalyticsEngine
    from src.reports import render_report

    # Use the engine to build the AnalyticsResult, then render.
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        engine = AnalyticsEngine()
        result = engine.run(sample_enriched_dataset,
                            output_path=Path(tmp) / "analytics.json")
    markdown = render_report(result)

    assert markdown.startswith("# Berlin AI Talent Radar")
    assert "Executive Summary" in markdown
    assert "Skill Landscape" in markdown
    assert "Architecture-Execution Spectrum" in markdown
    assert "EU AI Act Governance" in markdown
    assert "Company Intelligence" in markdown
    # Top skill should appear in the report.
    assert "`python`" in markdown
    # Governance gap rate appears in exec summary.
    assert "governance gaps" in markdown.lower()
