"""
Skill demand analytics.

Pure function: takes a list of ``EnrichedJob`` and returns a
``SkillAnalytics`` with skill frequency distributions, category
breakdowns, monthly trends (from HN-tagged data), and a coarse
growth classification (``explosive`` / ``steady`` / ``declining``).

GenAI vs traditional ML split is computed by intersecting each job's
flat skill list with two reference sets. A job that matches *any*
GenAI skill goes into the GenAI bucket; otherwise, a match against
the traditional ML set puts it in the traditional bucket. Jobs that
match neither are excluded from both counts.
"""

from collections import Counter, defaultdict

from src.models import EnrichedJob, SkillAnalytics
from src.utils.logger import get_logger

logger = get_logger("analytics.skill")

_GENAI_SKILLS = {
    "rag", "langchain", "llamaindex", "llm", "gpt_api",
    "prompt_engineering", "fine_tuning", "embeddings", "vector_database",
    "ai_agents", "openai", "anthropic_claude", "huggingface",
}
_TRADITIONAL_ML_SKILLS = {
    "pytorch", "tensorflow", "scikit_learn", "xgboost", "keras",
}


def compute_skill_analytics(jobs: list[EnrichedJob]) -> SkillAnalytics:
    """Compute skill demand analytics from enriched jobs."""
    total = len(jobs)
    if total == 0:
        return SkillAnalytics(total_jobs=0)

    logger.info("Computing skill analytics for %d jobs", total)

    skill_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    monthly_data: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int),
    )

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

        for category, skills in job.skills.items():
            category_counter[category] += len(skills)

        if flat_skills & _GENAI_SKILLS:
            genai_jobs += 1
        elif flat_skills & _TRADITIONAL_ML_SKILLS:
            traditional_ml_jobs += 1

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

        if job.hn_month:
            for skill in flat_skills:
                monthly_data[skill][job.hn_month] += 1

    top_20 = skill_counter.most_common(20)
    skill_pcts = {
        skill: round(count / total * 100, 1)
        for skill, count in skill_counter.items()
    }
    skill_growth = _classify_growth(monthly_data)

    result = SkillAnalytics(
        total_jobs=total,
        skill_counts=dict(skill_counter),
        skill_percentages=skill_pcts,
        category_counts=dict(category_counter),
        top_20_skills=top_20,
        monthly_trends={k: dict(v) for k, v in monthly_data.items()},
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
    Compare first-half vs second-half monthly totals to label growth.

    Returns ``"explosive"`` when the second half is ≥1.5× the first,
    ``"declining"`` when it's ≤0.6×, otherwise ``"steady"``. Skills
    with fewer than two distinct months default to ``"steady"`` —
    a single data point isn't a trend.
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
            continue

        ratio = second_half / first_half
        if ratio >= 1.5:
            growth[skill] = "explosive"
        elif ratio <= 0.6:
            growth[skill] = "declining"
        else:
            growth[skill] = "steady"
    return growth
