"""
Prompt templates for the Berlin AI Talent Radar RAG engine.

The system prompt defines the analyst persona (Big-4 consulting tone,
strict grounding in retrieved evidence). The user prompt formats the
retrieved chunks with structured headers so the model can cite
companies and EU AI Act article numbers in its answer.

Separating prompts from engine logic keeps both reviewable without
running the model.
"""

from src.models import SearchResult


SYSTEM_PROMPT = (
    "You are a senior AI labour-market analyst specialising in Berlin's "
    "technology sector and the EU AI Act (Regulation EU 2024/1689, "
    "enforceable 2 August 2026).\n\n"
    "Your task is to answer questions using ONLY the context provided "
    "below. Follow these rules:\n"
    "1. Base every claim on evidence from the context chunks.\n"
    "2. Cite sources: mention company names for job data, article "
    "numbers for EU AI Act references.\n"
    "3. If the context does not contain enough information to answer "
    "confidently, say so explicitly rather than speculating.\n"
    "4. Use precise numbers and percentages when the data supports them.\n"
    "5. Structure longer answers with bullet points or short paragraphs.\n"
    "6. Keep the tone professional and consultative."
)


def build_user_prompt(
    question: str,
    job_results: list[SearchResult],
    reg_results: list[SearchResult],
    query_type: str,
) -> str:
    """
    Format the user message with context chunks and the question.

    Args:
        question: Original user question.
        job_results: Retrieved job posting chunks.
        reg_results: Retrieved regulation chunks.
        query_type: ``"legal"`` | ``"market"`` | ``"mixed"``.

    Returns:
        Formatted user prompt string.
    """
    sections: list[str] = []

    if job_results:
        sections.append("=== JOB MARKET DATA ===")
        for i, r in enumerate(job_results, 1):
            meta = r.chunk.metadata
            header = (
                f"[Job {i}] {meta.company or 'Unknown'} — "
                f"{meta.title or 'Unknown'} "
                f"| Seniority: {meta.seniority or 'N/A'} "
                f"| Arch-Exec: {meta.arch_exec_score if meta.arch_exec_score is not None else 'N/A'} "
                f"| Skills: {', '.join(meta.skills) if meta.skills else 'N/A'}"
            )
            if meta.governance_gap:
                header += " | GOVERNANCE GAP"
            sections.append(header)
            sections.append(r.chunk.text)
            sections.append("")

    if reg_results:
        sections.append("=== EU AI ACT REFERENCES ===")
        for r in reg_results:
            meta = r.chunk.metadata
            header = (
                f"[Article {meta.article_number or '?'}] "
                f"{meta.article_title or 'Unknown'} "
                f"| Enforcement: {meta.enforcement_date or '2026-08-02'}"
            )
            if meta.penalty_reference:
                header += f" | Penalty: {meta.penalty_reference}"
            sections.append(header)
            sections.append(r.chunk.text)
            sections.append("")

    if query_type == "legal":
        sections.append(
            "Focus your answer on EU AI Act compliance and regulatory "
            "implications."
        )
    elif query_type == "market":
        sections.append(
            "Focus your answer on market trends, skill demand, and hiring "
            "patterns."
        )

    sections.append(f"QUESTION: {question}")
    return "\n".join(sections)
