"""
Company intelligence analytics.

Pure function: takes a list of ``EnrichedJob`` and returns a
``CompanyAnalytics`` with rankings by posting count, per-company
skill profiles (top 8), average arch-exec score, governance-gap flag,
and seniority mix.

Only the top ``top_count`` companies are profiled in detail —
beyond that the long tail of single-posting companies adds noise
without much signal.
"""

from collections import Counter, defaultdict
from typing import Any

from src.models import CompanyAnalytics, EnrichedJob
from src.utils.logger import get_logger

logger = get_logger("analytics.company")


def compute_company_analytics(
    jobs: list[EnrichedJob],
    top_count: int = 15,
) -> CompanyAnalytics:
    """Compute company intelligence analytics."""
    if not jobs:
        return CompanyAnalytics(total_companies=0)

    logger.info("Computing company analytics for %d jobs", len(jobs))

    company_jobs: dict[str, list[EnrichedJob]] = defaultdict(list)
    for j in jobs:
        company_jobs[j.company].append(j)

    total_companies = len(company_jobs)
    company_counts = Counter({c: len(js) for c, js in company_jobs.items()})
    top_companies = company_counts.most_common(top_count)

    rankings: list[dict[str, Any]] = []
    skill_profiles: dict[str, list[str]] = {}
    avg_arch_exec: dict[str, float] = {}
    governance_gaps: dict[str, bool] = {}
    seniority_mix: dict[str, dict[str, int]] = {}

    for company, count in top_companies:
        cjobs = company_jobs[company]

        rankings.append({
            "company": company,
            "count": count,
            "pct": round(count / len(jobs) * 100, 1),
        })

        skill_counter: Counter[str] = Counter()
        for j in cjobs:
            for skill in j.all_skills_flat:
                skill_counter[skill] += 1
        skill_profiles[company] = [s for s, _ in skill_counter.most_common(8)]

        scores = [j.arch_exec_score for j in cjobs]
        avg_arch_exec[company] = round(sum(scores) / len(scores), 3)

        governance_gaps[company] = any(
            j.eu_ai_act.governance_gap for j in cjobs
        )

        sen_counts: dict[str, int] = defaultdict(int)
        for j in cjobs:
            sen_counts[j.seniority] += 1
        seniority_mix[company] = dict(sen_counts)

    result = CompanyAnalytics(
        total_companies=total_companies,
        rankings=rankings,
        skill_profiles=skill_profiles,
        avg_arch_exec=avg_arch_exec,
        governance_gaps=governance_gaps,
        seniority_mix=seniority_mix,
    )

    logger.info(
        "Company analytics: %d total companies, top=%s (%d postings)",
        total_companies,
        top_companies[0][0] if top_companies else "N/A",
        top_companies[0][1] if top_companies else 0,
    )
    return result
