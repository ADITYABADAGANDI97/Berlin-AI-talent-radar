"""
Berlin AI Talent Radar — Streamlit Dashboard.

Five tabs read from the unified ``data/reports/analytics.json``
produced by ``src/analytics/engine.py``:

  1. Companies — rankings, skill profiles, governance-gap flags
  2. Skills — top 20, category breakdown, GenAI/remote/German rates
  3. Architecture-Execution Spectrum — histogram + seniority averages
  4. EU AI Act Governance — gap counts, high-risk domains, countdown
  5. RAG Chat — natural-language Q&A over the vector store

Run from the project root::

    streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Make project root importable so `from src.* import ...` works regardless
# of where streamlit is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import load_json  # noqa: E402


st.set_page_config(
    page_title="Berlin AI Talent Radar",
    page_icon="📡",
    layout="wide",
)


@st.cache_data
def load_analytics() -> dict | None:
    """Load ``data/reports/analytics.json`` once per session."""
    path = PROJECT_ROOT / "data" / "reports" / "analytics.json"
    if not path.exists():
        return None
    return load_json(path)


@st.cache_data
def load_enriched_jobs() -> list[dict]:
    """Load the per-posting enriched records — only needed by tabs that
    drill into individual postings (e.g. the governance-gap company list)."""
    path = PROJECT_ROOT / "data" / "processed" / "enriched_jobs.json"
    if not path.exists():
        return []
    return load_json(path)


# =============================================================================
# Tabs
# =============================================================================

def tab_companies(data: dict) -> None:
    """Tab 1: Company Intelligence."""
    import pandas as pd
    import plotly.express as px

    companies = data.get("companies", {}) or {}
    rankings = companies.get("rankings", []) or []

    st.header("Company Intelligence")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Companies", companies.get("total_companies", 0))
    with col2:
        if rankings:
            st.metric(
                "Top Employer",
                rankings[0]["company"],
                delta=f"{rankings[0]['count']} postings",
            )

    if rankings:
        df = pd.DataFrame(rankings)
        fig = px.bar(
            df, x="count", y="company", orientation="h",
            title="Top Companies by AI Job Postings",
            labels={"count": "Postings", "company": ""},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)

    skill_profiles = companies.get("skill_profiles", {}) or {}
    if skill_profiles:
        st.subheader("Company Skill Profiles (Top 8)")
        selected = st.selectbox("Select company", list(skill_profiles.keys()))
        if selected:
            st.write(", ".join(skill_profiles[selected]))

    gov_gaps = companies.get("governance_gaps", {}) or {}
    gap_companies = [c for c, has_gap in gov_gaps.items() if has_gap]
    if gap_companies:
        preview = ", ".join(gap_companies[:5])
        more = "..." if len(gap_companies) > 5 else ""
        st.warning(
            f"⚠️ {len(gap_companies)} companies have EU AI Act "
            f"governance gaps: {preview}{more}"
        )


def tab_skills(data: dict) -> None:
    """Tab 2: Skill Landscape."""
    import pandas as pd
    import plotly.express as px

    skills = data.get("skills", {}) or {}

    st.header("Skill Landscape")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Jobs", skills.get("total_jobs", 0))
    with col2:
        st.metric("GenAI Roles", f"{skills.get('genai_pct', 0)}%")
    with col3:
        st.metric("Remote/Hybrid", f"{skills.get('remote_pct', 0)}%")
    with col4:
        st.metric("German Required", f"{skills.get('german_pct', 0)}%")

    top_20 = skills.get("top_20_skills", []) or []
    if top_20:
        df = pd.DataFrame(top_20, columns=["skill", "count"])
        total = max(skills.get("total_jobs", 1), 1)
        df["pct"] = (df["count"] / total * 100).round(1)
        fig = px.bar(
            df, x="pct", y="skill", orientation="h",
            title="Top 20 Skills (% of postings)",
            labels={"pct": "% of Postings", "skill": ""},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
        st.plotly_chart(fig, use_container_width=True)

    categories = skills.get("category_counts", {}) or {}
    if categories:
        df = pd.DataFrame(
            [{"category": k, "mentions": v} for k, v in categories.items()]
        )
        fig = px.pie(
            df, values="mentions", names="category",
            title="Skill Categories Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)


def tab_arch_exec(data: dict) -> None:
    """Tab 3: Architecture-Execution Spectrum."""
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    ae = data.get("arch_exec", {}) or {}

    st.header("Architecture-Execution Spectrum")
    st.caption(
        "0.0 = Pure Execution (AI-vulnerable) → "
        "1.0 = Pure Architecture (Future-proof)"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Score", f"{ae.get('mean_score', 0):.2f}")
    with col2:
        st.metric(
            "Execution-Heavy", f"{ae.get('execution_heavy_pct', 0)}%",
            delta=f"{ae.get('execution_heavy_count', 0)} roles",
            delta_color="inverse",
        )
    with col3:
        st.metric(
            "Architecture-Heavy", f"{ae.get('architecture_heavy_pct', 0)}%",
            delta=f"{ae.get('architecture_heavy_count', 0)} roles",
        )

    bins = ae.get("histogram_bins", []) or []
    counts = ae.get("histogram_counts", []) or []
    if bins and counts:
        bin_labels = [
            f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(counts))
        ]
        colors = []
        for i in range(len(counts)):
            mid = (bins[i] + bins[i + 1]) / 2
            if mid < 0.40:
                colors.append("#ef4444")
            elif mid > 0.70:
                colors.append("#22c55e")
            else:
                colors.append("#3b82f6")
        fig = go.Figure(go.Bar(x=bin_labels, y=counts, marker_color=colors))
        fig.update_layout(
            title=(
                "Score Distribution "
                "(Red=Exec-Heavy, Blue=Balanced, Green=Arch-Heavy)"
            ),
            xaxis_title="Arch-Exec Score Range",
            yaxis_title="Number of Postings",
        )
        st.plotly_chart(fig, use_container_width=True)

    by_seniority = ae.get("by_seniority", {}) or {}
    if by_seniority:
        order = ["intern", "junior", "mid", "senior", "lead"]
        df = pd.DataFrame(
            [{"seniority": k, "score": v} for k, v in by_seniority.items()]
        )
        df["seniority"] = pd.Categorical(
            df["seniority"], categories=order, ordered=True,
        )
        df = df.sort_values("seniority")
        fig = px.bar(
            df, x="seniority", y="score",
            title="Average Arch-Exec Score by Seniority",
            labels={"score": "Mean Score", "seniority": ""},
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)


def tab_governance(data: dict) -> None:
    """Tab 4: EU AI Act Governance Gaps."""
    import pandas as pd
    import plotly.express as px

    gov = data.get("governance", {}) or {}

    st.header("EU AI Act Governance Gaps")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AI Roles", gov.get("total_ai_roles", 0))
    with col2:
        st.metric(
            "High-Risk Roles", gov.get("high_risk_count", 0),
            delta=f"{gov.get('high_risk_pct', 0)}% of AI roles",
        )
    with col3:
        st.metric(
            "Governance Gaps", gov.get("governance_gap_count", 0),
            delta=f"{gov.get('governance_gap_pct', 0)}%",
            delta_color="inverse",
        )

    days = gov.get("days_to_enforcement", 0)
    penalty = gov.get("max_penalty_eur", 35_000_000)
    if days > 0:
        st.error(
            f"⏰ **{days} days** until EU AI Act enforcement (2 August 2026). "
            f"Maximum penalty: **€{penalty:,.0f}** or 7% of global turnover."
        )

    by_domain = gov.get("by_domain", {}) or {}
    if by_domain:
        df = pd.DataFrame(
            [{"domain": k, "count": v} for k, v in by_domain.items()]
        )
        fig = px.bar(
            df, x="count", y="domain", orientation="h",
            title="High-Risk Domains (Annex III)",
            labels={"count": "Postings", "domain": ""},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Per-posting gap detail — articulates *where* each posting fails:
    # which Annex III domain triggers them, which EU AI Act articles
    # they're obliged to address, and which specific obligations they
    # did NOT demonstrate awareness of in the posting text.
    # ------------------------------------------------------------------
    st.subheader("⚠️ Where these postings fail to comply")
    st.caption(
        "Each posting below is an AI role in an EU AI Act Annex III "
        "high-risk domain whose public text addresses **none** of the "
        "governance obligations that apply to it. The breakdown names "
        "the specific articles and what each requires."
    )

    enriched = load_enriched_jobs()
    gap_postings = [
        j for j in enriched
        if j.get("eu_ai_act", {}).get("governance_gap")
    ]

    if not gap_postings:
        st.info("No governance gaps in the current dataset.")
    else:
        _render_gap_postings(gap_postings, gov.get("days_to_enforcement", 0))


# Maps EU AI Act article numbers → short label + the governance keywords
# that would signal awareness of that article's obligations. The article
# texts themselves live in config/eu_ai_act_articles.yaml; this is the
# detection-vocabulary projection of them used to articulate *what*
# compliance signal a posting failed to demonstrate.
_ARTICLE_OBLIGATIONS: dict[int, dict[str, object]] = {
    9: {
        "label": "Risk management system",
        "keywords": ["risk assessment", "risk management", "ai risk",
                     "iterative risk"],
    },
    10: {
        "label": "Data & data governance (incl. bias)",
        "keywords": ["bias", "data governance", "training data",
                     "data quality", "dataset documentation"],
    },
    11: {
        "label": "Technical documentation",
        "keywords": ["model card", "technical documentation",
                     "model documentation"],
    },
    13: {
        "label": "Transparency to deployers",
        "keywords": ["transparency", "explainability", "interpretable",
                     "interpretability"],
    },
    14: {
        "label": "Human oversight",
        "keywords": ["human oversight", "human-in-the-loop",
                     "human in the loop", "human review"],
    },
    15: {
        "label": "Accuracy, robustness, cybersecurity",
        "keywords": ["robustness", "model monitoring", "model drift",
                     "accuracy testing"],
    },
    26: {
        "label": "Deployer obligations",
        "keywords": ["compliance", "audit", "conformity assessment",
                     "responsible ai", "ai governance"],
    },
    50: {
        "label": "Transparency to natural persons",
        "keywords": ["informed", "user disclosure", "ai notice"],
    },
}


def _render_gap_postings(postings: list[dict], days_to_enforce: int) -> None:
    """Render one expander per gap-flagged posting with article-level detail."""
    # Group by domain so the user can scan a healthcare bucket, an
    # employment bucket, etc.
    by_domain: dict[str, list[dict]] = {}
    for p in postings:
        domains = p["eu_ai_act"].get("high_risk_domains") or ["(unspecified)"]
        for d in domains:
            by_domain.setdefault(d, []).append(p)

    for domain in sorted(by_domain.keys()):
        bucket = by_domain[domain]
        st.markdown(
            f"#### {domain.replace('_', ' ')} — {len(bucket)} posting(s)"
        )
        for p in bucket:
            title = p["title"] or "Untitled"
            company = p["company"] or "Unknown"
            articles = p["eu_ai_act"].get("relevant_articles") or []
            url = p.get("url", "")

            header = f"**{company}** — {title[:80]}"
            with st.expander(header):
                if url:
                    st.markdown(f"[Open original posting ↗]({url})")
                st.markdown(
                    f"**Source:** `{p.get('source', '?')}`  •  "
                    f"**Posted:** {(p.get('date_posted') or 'unknown')[:10]}"
                )
                st.markdown(
                    f"**Annex III domain:** {domain.replace('_', ' ')} "
                    f"→ triggers Articles "
                    f"**{', '.join(str(a) for a in articles)}**"
                )

                # Spell out what they failed to address.
                st.markdown("**Obligations not addressed in the posting:**")
                missing_rows = []
                for art in articles:
                    spec = _ARTICLE_OBLIGATIONS.get(art)
                    if not spec:
                        continue
                    expected = ", ".join(f"`{kw}`" for kw in spec["keywords"])
                    missing_rows.append(
                        f"- ❌ **Article {art} — {spec['label']}.** "
                        f"Posting contained none of: {expected}."
                    )
                if missing_rows:
                    st.markdown("\n".join(missing_rows))
                else:
                    st.markdown(
                        "_No article-level keyword catalogue is wired up "
                        "for this domain yet — only the headline gap is reported._"
                    )

    st.caption(
        f"All {len(postings)} flagged postings are real, live job ads — "
        "click any \"Open original posting\" link to verify. The "
        "absence of governance keywords in a public posting does not by "
        "itself prove non-compliance, but with "
        f"**{days_to_enforce} days** until enforcement (2 Aug 2026, "
        "max penalty €35M / 7% global turnover) it's a strong basis "
        "for vendor due-diligence questions."
    )


def tab_vendor_audit(data: dict) -> None:
    """
    Tab 5: Vendor Audit.

    Two action-oriented tools that sit on top of the same enriched
    dataset that powers the rest of the dashboard:

      1. **Compare companies side-by-side** — pick 2-4 employers from
         the dataset and see their AI-roles count, governance-gap
         count, top skills, seniority mix, and arch-exec mean in a
         single table. Useful when an HR / procurement team is
         choosing between two vendors that build AI for the same
         domain (e.g. two ATS providers, two credit-scoring engines).

      2. **Generate a vendor questionnaire** — pick a company (or
         filter to a domain) and the dashboard renders a markdown
         questionnaire built from the EU AI Act articles triggered by
         that company's postings. The output is copy-pasteable into
         an email or RFP — one section per Article, each with the
         specific compliance evidence to ask for.
    """
    import pandas as pd

    enriched = load_enriched_jobs()
    if not enriched:
        st.warning(
            "No enriched data found. Run "
            "`python main.py process` to enable this tab."
        )
        return

    st.header("Vendor Audit")
    st.caption(
        "Turn the dashboard data into procurement actions: compare "
        "potential vendors against each other, or generate a ready-to-send "
        "compliance questionnaire grounded in the EU AI Act articles their "
        "postings trigger."
    )

    audit_a, audit_b = st.tabs(
        ["🔬 Compare companies", "📋 Generate questionnaire"],
    )

    with audit_a:
        _render_company_compare(enriched, pd)

    with audit_b:
        _render_questionnaire(enriched, data.get("governance", {}) or {})


def _render_company_compare(enriched: list[dict], pd) -> None:
    """Side-by-side comparison panel."""
    by_company: dict[str, list[dict]] = {}
    for j in enriched:
        by_company.setdefault(j["company"], []).append(j)

    # Sort companies by AI role count so the dropdown's default order
    # surfaces the names that have the most signal.
    ordered = sorted(
        by_company.items(),
        key=lambda kv: sum(
            1 for j in kv[1] if j["eu_ai_act"]["is_ai_role"]
        ),
        reverse=True,
    )
    all_companies = [c for c, _ in ordered]

    default_picks = [c for c in all_companies if len(by_company[c]) > 1][:3]
    if not default_picks:
        default_picks = all_companies[:3]

    picks = st.multiselect(
        "Pick 2-4 companies to compare",
        options=all_companies,
        default=default_picks,
        max_selections=4,
    )

    if len(picks) < 2:
        st.info("Pick at least two companies to see the comparison.")
        return

    rows: list[dict[str, object]] = []
    for company in picks:
        jobs = by_company[company]
        ai = [j for j in jobs if j["eu_ai_act"]["is_ai_role"]]
        high_risk = [
            j for j in ai
            if j["eu_ai_act"]["touches_high_risk_domain"]
        ]
        gaps = [j for j in high_risk if j["eu_ai_act"]["governance_gap"]]

        # Top 5 skills across all postings
        from collections import Counter
        skill_counter: Counter = Counter()
        for j in jobs:
            for s in j.get("all_skills_flat", []):
                skill_counter[s] += 1
        top_skills = ", ".join(s for s, _ in skill_counter.most_common(5))

        # Mean arch-exec
        scores = [j["arch_exec_score"] for j in jobs]
        mean_arch = (
            f"{sum(scores) / len(scores):.2f}" if scores else "—"
        )

        # Seniority mix
        sen_counter: Counter = Counter(j["seniority"] for j in jobs)
        sen_mix = ", ".join(
            f"{lvl}:{n}" for lvl, n in sen_counter.most_common()
        )

        # Annex III domains touched (deduped across postings)
        domains: set[str] = set()
        for j in high_risk:
            domains.update(j["eu_ai_act"]["high_risk_domains"])

        # Governance keywords ever mentioned (across all postings)
        gov_kw: set[str] = set()
        for j in jobs:
            gov_kw.update(j["eu_ai_act"]["governance_keywords_found"])

        rows.append({
            "Metric": "Company",
            company: company,
        })
        rows.append({
            "Metric": "Total postings", company: len(jobs),
        })
        rows.append({
            "Metric": "AI roles", company: len(ai),
        })
        rows.append({
            "Metric": "High-risk AI roles (Annex III)",
            company: len(high_risk),
        })
        rows.append({
            "Metric": "Governance gaps", company: len(gaps),
        })
        rows.append({
            "Metric": "Annex III domains",
            company: ", ".join(sorted(domains)) or "—",
        })
        rows.append({
            "Metric": "Mean arch-exec score", company: mean_arch,
        })
        rows.append({
            "Metric": "Top 5 skills",
            company: top_skills or "—",
        })
        rows.append({
            "Metric": "Seniority mix", company: sen_mix or "—",
        })
        rows.append({
            "Metric": "Governance keywords mentioned",
            company: ", ".join(sorted(gov_kw)) or "(none)",
        })

    # Pivot so each company is a column.
    df = pd.DataFrame(rows).groupby("Metric", sort=False).first().reset_index()
    df = df[["Metric"] + picks]
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Highlight which picks have governance gaps as a quick verdict line.
    gap_status = []
    for c in picks:
        n_gaps = sum(
            1 for j in by_company[c]
            if j["eu_ai_act"]["governance_gap"]
        )
        if n_gaps:
            gap_status.append(f"⚠️ **{c}** — {n_gaps} gap posting(s)")
        else:
            gap_status.append(f"✅ **{c}** — no gap postings in dataset")
    st.markdown("**Compliance signal:**  •  " + "  •  ".join(gap_status))


def _render_questionnaire(enriched: list[dict], gov_summary: dict) -> None:
    """Markdown questionnaire generator panel."""
    # Build the universe of companies that have at least one gap — those
    # are the ones a procurement team would actually want to question.
    gap_companies = sorted({
        j["company"] for j in enriched
        if j["eu_ai_act"]["governance_gap"]
    })

    if not gap_companies:
        st.info(
            "No companies with governance gaps in the current dataset. "
            "Refresh with `python main.py collect && python main.py process` "
            "to pull a fresh sample."
        )
        return

    company = st.selectbox(
        "Pick a company to generate a questionnaire for",
        options=gap_companies,
        index=0,
    )

    # Gather all gap postings for this company so the questionnaire
    # covers every Annex III domain they touch.
    postings = [
        j for j in enriched
        if j["company"] == company and j["eu_ai_act"]["governance_gap"]
    ]
    domains = sorted({
        d for p in postings
        for d in p["eu_ai_act"]["high_risk_domains"]
    })
    articles = sorted({
        a for p in postings
        for a in p["eu_ai_act"]["relevant_articles"]
    })

    days = gov_summary.get("days_to_enforcement", 0)
    penalty = gov_summary.get("max_penalty_eur", 35_000_000)

    md = _build_questionnaire_markdown(
        company=company,
        postings=postings,
        domains=domains,
        articles=articles,
        days_to_enforce=days,
        max_penalty_eur=penalty,
    )

    st.markdown("---")
    st.markdown(md)
    st.markdown("---")
    st.download_button(
        label="⬇️ Download questionnaire (.md)",
        data=md,
        file_name=f"{company.replace(' ', '_')}_eu_ai_act_audit.md",
        mime="text/markdown",
    )


def _build_questionnaire_markdown(
    company: str,
    postings: list[dict],
    domains: list[str],
    articles: list[int],
    days_to_enforce: int,
    max_penalty_eur: int,
) -> str:
    """Compose a markdown questionnaire grounded in the relevant Articles."""
    lines: list[str] = []
    lines.append(f"# EU AI Act Compliance Questionnaire — {company}")
    lines.append("")
    lines.append(
        f"**Context.** {company} has {len(postings)} public job posting(s) "
        f"in our dataset that describe building or deploying AI in "
        f"{', '.join(d.replace('_', ' ') for d in domains)} — a high-risk "
        f"domain under Annex III of the EU AI Act "
        f"(Regulation EU 2024/1689). The postings do not address the "
        f"governance obligations that apply. With **{days_to_enforce} "
        f"days** until enforcement (2 August 2026, max penalty "
        f"€{max_penalty_eur:,} / 7% global turnover), the following "
        f"questions ask for the compliance evidence that should exist "
        f"regardless of whether it appears in recruiting material."
    )
    lines.append("")
    lines.append("## Postings referenced")
    lines.append("")
    for p in postings:
        title = p.get("title", "Untitled")
        url = p.get("url", "")
        if url:
            lines.append(f"- [{title}]({url})")
        else:
            lines.append(f"- {title}")
    lines.append("")

    if not articles:
        lines.append(
            "_No specific EU AI Act articles were derivable from the "
            "domain match. Treat the headline gap as the only signal._"
        )
        return "\n".join(lines)

    for art in articles:
        spec = _ARTICLE_OBLIGATIONS.get(art)
        if not spec:
            continue
        lines.append(f"## Article {art} — {spec['label']}")
        lines.append("")
        # Compose 2-3 audit questions per article. Keep these
        # vendor-facing: each question should be answerable with a
        # document, a process description, or an artefact.
        questions = _questions_for_article(art)
        for q in questions:
            lines.append(f"- [ ] {q}")
        lines.append("")
        lines.append(
            f"_Public posting did not mention any of: "
            f"{', '.join('`' + k + '`' for k in spec['keywords'])}._"
        )
        lines.append("")

    lines.append("---")
    lines.append(
        "_Questionnaire generated automatically from job-posting analysis. "
        "Absence of a governance keyword in a posting is a signal, not "
        "proof of non-compliance — but every question above is one a "
        "deployer is obliged to answer under the cited Article."
    )
    return "\n".join(lines)


def _questions_for_article(article: int) -> list[str]:
    """Concrete audit questions per Article. Vendor-answerable, not vague."""
    catalogue = {
        9: [
            "Provide the documented risk management process for this "
            "AI system covering its full lifecycle (design, development, "
            "deployment, monitoring).",
            "Show the most recent risk assessment and the residual risks "
            "you accepted, with the rationale.",
            "Describe how the risk management system is reviewed and "
            "updated when material changes occur.",
        ],
        10: [
            "Provide a description of the training, validation and "
            "test data sets — sources, collection periods, and "
            "preparation steps.",
            "What measures detect, prevent and mitigate bias in the "
            "data and the resulting model output?",
            "How is data quality monitored over time, including drift "
            "and representativity of the deployment population?",
        ],
        11: [
            "Provide the technical documentation that demonstrates "
            "conformity with Articles 9-15 — including model "
            "architecture, training methodology, and performance "
            "metrics.",
            "Provide the most recent version of the model card / "
            "model documentation shared with deployers.",
        ],
        13: [
            "Describe the information you provide to deployers so they "
            "can interpret model output correctly and understand its "
            "limits.",
            "What's your communicated guidance on intended use, "
            "foreseeable misuse, and known performance limitations?",
        ],
        14: [
            "Describe the human-oversight mechanisms available to "
            "deployers — including the controls a human operator can "
            "use to intervene, override, or disable the system.",
            "Who at the deployer organisation is identified as the "
            "responsible human overseer, and what training are they "
            "given?",
            "How are oversight events (intervention, override) logged "
            "and reviewed?",
        ],
        15: [
            "Provide the published accuracy, robustness and "
            "cybersecurity metrics for the model, with the test "
            "methodology used.",
            "How is the model resilient to data poisoning, model "
            "evasion, and adversarial inputs?",
            "What's the monitoring posture for accuracy drift in "
            "production, and what triggers retraining?",
        ],
        26: [
            "Confirm conformity assessment status (self-assessment "
            "vs. third-party, notified body involved).",
            "Provide the CE marking documentation and the registration "
            "entry in the EU database (Article 49).",
            "Provide the deployer-facing instructions for use "
            "(Article 13(2)).",
        ],
        50: [
            "Describe how natural persons interacting with the system "
            "are informed they are interacting with an AI system.",
            "If the system generates synthetic content, describe the "
            "machine-readable marking applied.",
        ],
    }
    return catalogue.get(article, [
        f"Provide the compliance evidence required under Article {article}.",
    ])


def tab_rag_chat(data: dict) -> None:
    """Tab 5: RAG Chat Interface."""
    st.header("Ask the AI Market Analyst")
    st.caption(
        "Ask questions about Berlin's AI job market or EU AI Act compliance. "
        "Powered by RAG over the full dataset."
    )

    embeddings_dir = PROJECT_ROOT / "data" / "embeddings"
    has_vectors = embeddings_dir.exists() and any(embeddings_dir.glob("*.npz"))
    if not has_vectors:
        st.warning(
            "Vector store not built yet. Run `python main.py embed` first "
            "to enable the RAG chat (requires OPENAI_API_KEY)."
        )
        return

    question = st.text_input(
        "Your question:",
        placeholder=(
            "e.g., What skills are most in demand for ML engineers in Berlin?"
        ),
    )

    if st.button("Ask", type="primary") and question:
        with st.spinner("Thinking..."):
            try:
                from src.embeddings import Embedder
                from src.rag import RAGEngine
                from src.storage import NumpyVectorStore

                store = NumpyVectorStore()
                store.load()
                embedder = Embedder()
                engine = RAGEngine(store, embedder)
                result = engine.query(question)

                st.markdown(f"**Confidence:** {result.confidence}")
                st.markdown(result.answer)

                if result.sources_jobs:
                    st.markdown(
                        f"**Job sources:** {', '.join(result.sources_jobs)}"
                    )
                if result.sources_legal:
                    st.markdown(
                        f"**Legal sources:** {', '.join(result.sources_legal)}"
                    )

                with st.expander("Confidence Breakdown"):
                    for signal, score in result.confidence_scores.items():
                        st.progress(score, text=f"{signal}: {score:.2f}")
            except Exception as exc:
                st.error(f"Error: {exc}")

    st.subheader("Example Questions")
    for example in [
        "What are the top 5 skills for RAG engineers in Berlin?",
        "Which companies have EU AI Act governance gaps?",
        "How does seniority correlate with architecture vs execution roles?",
        "What does Article 14 of the EU AI Act require for human oversight?",
        "Is Python or TypeScript more in demand for AI roles?",
    ]:
        st.code(example, language=None)


def main() -> None:
    """Dashboard entry point."""
    st.title("📡 Berlin AI Talent Radar")
    st.caption("AI Job Market Intelligence × EU AI Act Compliance Analysis")

    analytics = load_analytics()
    if analytics is None:
        st.warning(
            "No analytics data found. Run the pipeline first:\n\n"
            "```bash\n"
            "python main.py demo    # No API keys required\n"
            "python main.py full    # Full pipeline (needs API keys)\n"
            "```"
        )
        return

    tabs = st.tabs([
        "🏢 Companies",
        "🛠️ Skills",
        "⚖️ Arch-Exec Spectrum",
        "🇪🇺 Governance Gaps",
        "🛡️ Vendor Audit",
        "💬 RAG Chat",
    ])
    with tabs[0]:
        tab_companies(analytics)
    with tabs[1]:
        tab_skills(analytics)
    with tabs[2]:
        tab_arch_exec(analytics)
    with tabs[3]:
        tab_governance(analytics)
    with tabs[4]:
        tab_vendor_audit(analytics)
    with tabs[5]:
        tab_rag_chat(analytics)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pipeline Status**")
    st.sidebar.metric("Total Jobs", analytics.get("total_jobs", 0))
    st.sidebar.metric("Data Sources", len(analytics.get("data_sources", [])))
    date_range = analytics.get("date_range", {}) or {}
    if date_range:
        st.sidebar.caption(
            f"{date_range.get('earliest', '?')} → "
            f"{date_range.get('latest', '?')}"
        )
    cost = analytics.get("cost_summary", {}) or {}
    if cost.get("total_eur") is not None:
        st.sidebar.metric(
            "Pipeline Cost", f"€{cost['total_eur']:.4f}",
            delta=f"budget €{cost.get('budget_eur', 30.0):.2f}",
        )


if __name__ == "__main__":
    main()
