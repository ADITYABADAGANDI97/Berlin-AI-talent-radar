"""
Tests for src/processors/.

Cover the enrichment pipeline end-to-end on a handful of synthetic
``RawJob`` records: cleaner output, skill extraction, governance gap
logic, seniority + German + remote detection, and cross-source
fuzzy deduplication.
"""

from __future__ import annotations

import pytest


def test_pipeline_enriches_and_dedupes(make_raw):
    from src.processors import EnrichmentPipeline

    raw = [
        make_raw(
            "Acme Bank GmbH",
            "Senior ML Engineer, Credit Scoring",
            "<p>Senior ML engineer to build credit scoring and loan approval "
            "models. Own the system design and architecture. Mentor junior "
            "engineers. Drive technical roadmap. 5+ years experience with "
            "Python, PyTorch, SQL, Docker.</p>" + " padding text" * 20,
            "jsearch_001",
        ),
        # Near-duplicate from a different source — should merge into the
        # longer JSearch posting.
        make_raw(
            "Acme Bank",
            "Senior ML Engineer - Credit Scoring",
            "Senior ML engineer at Acme Bank focused on credit scoring and "
            "loan approval models. " + "padding " * 30,
            "hn_001",
            source="hackernews",
        ),
        make_raw(
            "Werkstudent GmbH",
            "Werkstudent Data Analyst (Praktikum)",
            "Werkstudentenstelle. Du baust Dashboards und schreibst SQL "
            "Queries. Fließendes Deutsch erforderlich. Hands-on coding "
            "daily." + " mehr Text" * 30,
            "bsj_001",
            source="bsj",
        ),
    ]

    pipeline = EnrichmentPipeline()
    enriched = pipeline.enrich(raw)

    assert len(enriched) == 2, "near-duplicates should merge cross-source"
    companies = {j.company for j in enriched}
    assert "Werkstudent GmbH" in companies

    bank = next(j for j in enriched if "acme" in j.company.lower())
    assert bank.eu_ai_act.is_ai_role is True
    assert bank.eu_ai_act.touches_high_risk_domain is True
    assert "Essential_Services" in bank.eu_ai_act.high_risk_domains
    assert bank.eu_ai_act.governance_gap is True
    assert bank.seniority == "senior"
    assert bank.source_id == "jsearch_001", (
        "deduper should keep the posting with the longer description"
    )

    werk = next(j for j in enriched if "werkstudent" in j.company.lower())
    assert werk.seniority == "intern"
    assert werk.requires_german is True
    assert werk.eu_ai_act.is_ai_role is False


def test_governance_gap_does_not_fire_when_keywords_present(make_raw):
    """A posting with human oversight + bias audit must NOT be a gap."""
    from src.processors import EnrichmentPipeline

    raw = [make_raw(
        "Bio Health AG",
        "Data Scientist, Talent Intelligence",
        "Resume screening and candidate screening AI. We run regular bias "
        "audits with human oversight and model documentation. "
        "Required: Python, NLP." + " " * 1 + "additional context " * 30,
        "arbeitnow_001",
        source="arbeitnow",
    )]

    enriched = EnrichmentPipeline().enrich(raw)
    assert len(enriched) == 1
    bio = enriched[0]
    assert bio.eu_ai_act.is_ai_role is True
    assert bio.eu_ai_act.touches_high_risk_domain is True
    assert "Employment" in bio.eu_ai_act.high_risk_domains
    assert bio.eu_ai_act.governance_gap is False
    assert bio.eu_ai_act.governance_keyword_count > 0


def test_arch_exec_score_separates_principal_from_junior(make_raw):
    from src.processors import EnrichmentPipeline

    principal = make_raw(
        "BigCo", "Principal Engineer, Platform",
        "Lead the technical vision. Drive technical roadmap. Make "
        "architectural trade-offs at scale. Mentor staff engineers. "
        "Cross-functional stakeholder management. Define standards. "
        "Influence engineering culture. " * 5,
        "j1",
    )
    junior = make_raw(
        "BigCo", "Junior Frontend Developer",
        "Build user interfaces. Write code daily. Implement features. "
        "Debug bugs. Follow established patterns. Daily standups. "
        "0-2 years experience. " * 5,
        "j2",
    )
    enriched = EnrichmentPipeline().enrich([principal, junior])
    by_seniority = {j.seniority: j for j in enriched}
    assert by_seniority["lead"].arch_exec_score > 0.7
    assert by_seniority["junior"].arch_exec_score < 0.4


def test_pipeline_skips_short_descriptions(make_raw):
    """Cleaner drops postings whose cleaned text is < min_description_length."""
    from src.processors import EnrichmentPipeline

    raw = [
        make_raw("OK Co", "Engineer", "x" * 200 + " job description", "ok1"),
        make_raw("Too Short", "Engineer", "tiny job desc", "ts1"),
    ]
    enriched = EnrichmentPipeline().enrich(raw)
    companies = {j.company for j in enriched}
    assert "OK Co" in companies
    assert "Too Short" not in companies
