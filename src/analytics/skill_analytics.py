"""
Skill demand analytics for Stage 5.

Computes skill frequency distributions, category breakdowns, monthly
trends (from HN data), and growth classifications from enriched jobs.

Usage::

    from src.analytics.skill_analytics import compute_skill_analytics
    result = compute_skill_analytics(enriched_jobs)
"""

from collections import Counter, defaultdict

from src.models import EnrichedJob, SkillAnalytics
from src.utils.logger import get_logger

logger = get_logger("analytics.skill")

# LLM/GenAI category skills for GenAI vs traditional ML classification
_GENAI_SKILLS = {
    "rag", "langchain", "llamaindex", "llm", "gpt_api",
    "prompt_engineering", "fine_tuning", "embeddings", "vector_database",
    "ai_agents", "openai", "anthropic_claude", "huggingface",
}


def compute_skill_analytics(jobs: list[EnrichedJob]) -> SkillAnalytics:
    """
    Compute skill demand analytics from enriched jobs.

    Args:
        jobs: List of EnrichedJob objects with skill extraction complete.

    Returns:
        SkillAnalytics with counts, percentages, trends, and growth.
    """
    total = len(jobs)
    if total == 0:
        return SkillAnalytics(total_jobs=0)

    logger.info("Computing skill analytics for %d jobs", total)

    # Skill counts (how many jobs mention each skill)
    skill_counter: Counter = Counter()
    category_counter: Counter = Counter()
    monthly_data: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    genai_jobs = 0
    traditional_ml_jobs = 0
    docker_jobs = 0
    phd_jobs = 0
    masters_jobs = 0
    german_jobs = 0
    remote_jobs = 0

    for job in jobs:
        flat_skills = set(job.all_skills_flat)
        for skill in flat_skills:
            skill_counter[skill] += 1

        # Category counts
        for category, skills in job.skills.items():
            category_counter[category] += len(skills)

        # GenAI vs traditional ML
        if flat_skills & _GENAI_SKILLS:
            genai_jobs += 1
        elif flat_skills & {"pytorch", "tensorflow", "scikit_learn", "xgboost", "keras"}:
            traditional_ml_jobs += 1

        # Specific skill flags
        if "docker" in flat_skills:
            docker_jobs += 1
        if "phd" in flat_skills:
            phd_jobs += 1
        if "masters" in flat_skills:
            masters_jobs += 1
        if job.requires_german:
            german_jobs += 1
        if job.is_remote:
            remote_jobs += 1

        # Monthly trends (HN data only)
        if job.hn_month:
            for skill in flat_skills:
                monthly_data[skill][job.hn_month] += 1

    # Top 20 skills
    top_20 = skill_counter.most_common(20)

    # Percentages
    skill_pcts = {
        skill: round(count / total * 100, 1)
        for skill, count in skill_counter.items()
    }

    # Skill growth classification based on monthly trends
    skill_growth = _classify_growth(monthly_data)

    result = SkillAnalytics(
        total_jobs=total,
        skill_counts=dict(skill_counter),
        skill_percentages=skill_pcts,
        category_counts=dict(category_counter),
        top_20_skills=top_20,
        monthly_trends=dict(monthly_data),
        skill_growth=skill_growth,
        genai_count=genai_jobs,
        traditional_ml_count=traditional_ml_jobs,
        docker_count=docker_jobs,
        phd_count=phd_jobs,
        masters_count=masters_jobs,
        german_required_count=german_jobs,
        remote_count=remote_jobs,
        german_pct=round(german_jobs / total * 100, 1),
        remote_pct=round(remote_jobs / total * 100, 1),
        genai_pct=round(genai_jobs / total * 100, 1),
        traditional_ml_pct=round(traditional_ml_jobs / total * 100, 1),
    )

    logger.info(
        "Skill analytics: top skill=%s (%d%%), GenAI=%d%%, remote=%d%%",
        top_20[0][0] if top_20 else "N/A",
        round(top_20[0][1] / total * 100) if top_20 else 0,
        round(genai_jobs / total * 100),
        round(remote_jobs / total * 100),
    )
    return result


def _classify_growth(
    monthly_data: dict[str, dict[str, int]],
) -> dict[str, str]:
    """
    Classify skill growth as explosive, steady, or declining.

    Uses simple comparison of first-half vs second-half monthly totals.
    """
    growth: dict[str, str] = {}

    for skill, months in monthly_data.items():
        if len(months) < 2:
            growth[skill] = "steady"
            continue

        sorted_months = sorted(months.keys())
        mid = len(sorted_months) // 2
        first_half = sum(months[m] for m in sorted_months[:mid])
        second_half = sum(months[m] for m in sorted_months[mid:])

        if first_half == 0:
            growth[skill] = "explosive" if second_half > 0 else "steady"
        else:
            ratio = second_half / first_half
            if ratio >= 1.5:
                growth[skill] = "explosive"
            elif ratio <= 0.6:
                growth[skill] = "declining"
            else:
                growth[skill] = "steady"

    return growth
