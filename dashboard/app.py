"""
Berlin AI Talent Radar — Streamlit Dashboard
==============================================
Interactive analytics dashboard with 5 tabs:
1. Company Intelligence
2. Skill Landscape
3. Architecture-Execution Spectrum
4. EU AI Act Governance Gaps
5. RAG Chat Interface

Run::

    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.Io import load_json

# Page config
st.set_page_config(
    page_title="Berlin AI Talent Radar",
    page_icon="📡",
    layout="wide",
)


@st.cache_data
def load_analytics():
    """Load analytics result from disk."""
    path = PROJECT_ROOT / "data" / "reports" / "analytics.json"
    if not path.exists():
        return None
    return load_json(path)


def tab_companies(data: dict) -> None:
    """Tab 1: Company Intelligence."""
    companies = data.get("companies", {})
    rankings = companies.get("rankings", [])

    st.header("Company Intelligence")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Companies", companies.get("total_companies", 0))

    with col2:
        if rankings:
            st.metric("Top Employer", rankings[0]["company"],
                      delta=f"{rankings[0]['count']} postings")

    if rankings:
        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame(rankings)
        fig = px.bar(
            df, x="count", y="company", orientation="h",
            title="Top Companies by AI Job Postings",
            labels={"count": "Postings", "company": ""},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Skill profiles
    skill_profiles = companies.get("skill_profiles", {})
    if skill_profiles:
        st.subheader("Company Skill Profiles (Top 8)")
        selected = st.selectbox("Select company", list(skill_profiles.keys()))
        if selected:
            st.write(", ".join(skill_profiles[selected]))

    # Governance gaps
    gov_gaps = companies.get("governance_gaps", {})
    if gov_gaps:
        gap_companies = [c for c, has_gap in gov_gaps.items() if has_gap]
        if gap_companies:
            st.warning(
                f"⚠️ {len(gap_companies)} companies have EU AI Act governance gaps: "
                f"{', '.join(gap_companies[:5])}{'...' if len(gap_companies) > 5 else ''}"
            )


def tab_skills(data: dict) -> None:
    """Tab 2: Skill Landscape."""
    skills = data.get("skills", {})

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

    # Top 20 skills chart
    top_20 = skills.get("top_20_skills", [])
    if top_20:
        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame(top_20, columns=["skill", "count"])
        total = skills.get("total_jobs", 1)
        df["pct"] = (df["count"] / total * 100).round(1)

        fig = px.bar(
            df, x="pct", y="skill", orientation="h",
            title="Top 20 Skills (% of postings)",
            labels={"pct": "% of Postings", "skill": ""},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
        st.plotly_chart(fig, use_container_width=True)

    # Category breakdown
    categories = skills.get("category_counts", {})
    if categories:
        import plotly.express as px
        import pandas as pd

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
    ae = data.get("arch_exec", {})

    st.header("Architecture-Execution Spectrum")
    st.caption(
        "0.0 = Pure Execution (AI-vulnerable) → 1.0 = Pure Architecture (Future-proof)"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Score", f"{ae.get('mean_score', 0):.2f}")
    with col2:
        st.metric("Execution-Heavy", f"{ae.get('execution_heavy_pct', 0)}%",
                  delta=f"{ae.get('execution_heavy_count', 0)} roles",
                  delta_color="inverse")
    with col3:
        st.metric("Architecture-Heavy", f"{ae.get('architecture_heavy_pct', 0)}%",
                  delta=f"{ae.get('architecture_heavy_count', 0)} roles")

    # Histogram
    bins = ae.get("histogram_bins", [])
    counts = ae.get("histogram_counts", [])
    if bins and counts:
        import plotly.graph_objects as go

        bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(counts))]
        colors = []
        for i in range(len(counts)):
            mid = (bins[i] + bins[i + 1]) / 2
            if mid < 0.40:
                colors.append("#ef4444")  # red
            elif mid > 0.70:
                colors.append("#22c55e")  # green
            else:
                colors.append("#3b82f6")  # blue

        fig = go.Figure(go.Bar(
            x=bin_labels, y=counts, marker_color=colors,
        ))
        fig.update_layout(
            title="Score Distribution (Red=Exec-Heavy, Blue=Balanced, Green=Arch-Heavy)",
            xaxis_title="Arch-Exec Score Range",
            yaxis_title="Number of Postings",
        )
        st.plotly_chart(fig, use_container_width=True)

    # By seniority
    by_seniority = ae.get("by_seniority", {})
    if by_seniority:
        import plotly.express as px
        import pandas as pd

        order = ["intern", "junior", "mid", "senior", "lead"]
        df = pd.DataFrame([
            {"seniority": k, "score": v}
            for k, v in by_seniority.items()
        ])
        df["seniority"] = pd.Categorical(df["seniority"], categories=order, ordered=True)
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
    gov = data.get("governance", {})

    st.header("EU AI Act Governance Gaps")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AI Roles", gov.get("total_ai_roles", 0))
    with col2:
        st.metric("High-Risk Roles", gov.get("high_risk_count", 0),
                  delta=f"{gov.get('high_risk_pct', 0)}% of AI roles")
    with col3:
        st.metric("Governance Gaps", gov.get("governance_gap_count", 0),
                  delta=f"{gov.get('governance_gap_pct', 0)}%",
                  delta_color="inverse")

    # Enforcement countdown
    days = gov.get("days_to_enforcement", 0)
    penalty = gov.get("max_penalty_eur", 35_000_000)
    if days > 0:
        st.error(
            f"⏰ **{days} days** until EU AI Act enforcement (2 August 2026). "
            f"Maximum penalty: **€{penalty:,.0f}** or 7% of global turnover."
        )

    # Domain breakdown
    by_domain = gov.get("by_domain", {})
    if by_domain:
        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame([
            {"domain": k, "count": v} for k, v in by_domain.items()
        ])
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

    # Check if vector store exists
    embeddings_dir = PROJECT_ROOT / "data" / "embeddings"
    if not embeddings_dir.exists() or not list(embeddings_dir.glob("*.npz")):
        st.warning(
            "Vector store not built yet. Run `python main.py embed` first "
            "to enable the RAG chat."
        )
        return

    # Chat interface
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What skills are most in demand for ML engineers in Berlin?",
    )

    if st.button("Ask", type="primary") and question:
        with st.spinner("Thinking..."):
            try:
                from src.embeddings import Embedder
                from src.storage import NumpyVectorStore
                from src.rag import RAGEngine

                store = NumpyVectorStore()
                store.load()
                embedder = Embedder()
                engine = RAGEngine(store, embedder)
                result = engine.query(question)

                # Display result
                st.markdown(f"**Confidence:** {result.confidence}")
                st.markdown(result.answer)

                if result.sources_jobs:
                    st.markdown(f"**Job sources:** {', '.join(result.sources_jobs)}")
                if result.sources_legal:
                    st.markdown(f"**Legal sources:** {', '.join(result.sources_legal)}")

                # Confidence breakdown
                with st.expander("Confidence Breakdown"):
                    for signal, score in result.confidence_scores.items():
                        st.progress(score, text=f"{signal}: {score:.2f}")

            except Exception as exc:
                st.error(f"Error: {exc}")

    # Example questions
    st.subheader("Example Questions")
    examples = [
        "What are the top 5 skills for RAG engineers in Berlin?",
        "Which companies have EU AI Act governance gaps?",
        "How does seniority correlate with architecture vs execution roles?",
        "What does Article 14 of the EU AI Act require for human oversight?",
        "Is Python or TypeScript more in demand for AI roles?",
    ]
    for ex in examples:
        st.code(ex, language=None)


def main() -> None:
    """Main dashboard entry point."""
    st.title("📡 Berlin AI Talent Radar")
    st.caption("AI Job Market Intelligence × EU AI Act Compliance Analysis")

    # Load analytics data
    analytics = load_analytics()

    if analytics is None:
        st.warning(
            "No analytics data found. Run the pipeline first:\n\n"
            "```bash\n"
            "python main.py full    # Full pipeline\n"
            "python main.py demo    # Demo with sample data\n"
            "```"
        )
        return

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏢 Companies",
        "🛠️ Skills",
        "⚖️ Arch-Exec Spectrum",
        "🇪🇺 Governance Gaps",
        "💬 RAG Chat",
    ])

    with tab1:
        tab_companies(analytics)
    with tab2:
        tab_skills(analytics)
    with tab3:
        tab_arch_exec(analytics)
    with tab4:
        tab_governance(analytics)
    with tab5:
        tab_rag_chat(analytics)

    # Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pipeline Status**")
    st.sidebar.metric("Total Jobs", analytics.get("total_jobs", 0))
    st.sidebar.metric("Data Sources", len(analytics.get("data_sources", [])))
    date_range = analytics.get("date_range", {})
    if date_range:
        st.sidebar.caption(
            f"{date_range.get('earliest', '?')} → {date_range.get('latest', '?')}"
        )


if __name__ == "__main__":
    main()
