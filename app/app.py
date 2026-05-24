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
