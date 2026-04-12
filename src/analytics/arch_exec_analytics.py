"""
Architecture-Execution Spectrum analytics for Stage 5.

Computes score distributions, seniority correlations, company profiles,
histograms, and identifies the most architecture-heavy and execution-heavy
postings.

Usage::

    from src.analytics.arch_exec_analytics import compute_arch_exec_analytics
    result = compute_arch_exec_analytics(enriched_jobs)
"""

from collections import defaultdict
from typing import Any

import numpy as np

from src.models import ArchExecAnalytics, EnrichedJob
from src.utils.logger import get_logger

logger = get_logger("analytics.arch_exec")


def compute_arch_exec_analytics(
    jobs: list[EnrichedJob],
    execution_threshold: float = 0.40,
    architecture_threshold: float = 0.70,
    top_companies_count: int = 15,
    bins: int = 10,
) -> ArchExecAnalytics:
    """
    Compute Architecture-Execution Spectrum analytics.

    Args:
        jobs: List of EnrichedJob objects with arch-exec scores.
        execution_threshold: Below this = execution-heavy.
        architecture_threshold: Above this = architecture-heavy.
        top_companies_count: Number of companies for by_company.
        bins: Number of histogram bins.

    Returns:
        ArchExecAnalytics with distributions and correlations.
    """
    total = len(jobs)
    if total == 0:
        return ArchExecAnalytics(
            total_scored=0, mean_score=0.0, median_score=0.0, std_score=0.0,
            execution_heavy_count=0, execution_heavy_pct=0.0,
            architecture_heavy_count=0, architecture_heavy_pct=0.0,
            balanced_count=0,
        )

    logger.info("Computing arch-exec analytics for %d jobs", total)

    scores = np.array([j.arch_exec_score for j in jobs])

    # Basic statistics
    mean_score = float(np.mean(scores))
    median_score = float(np.median(scores))
    std_score = float(np.std(scores))

    # Distribution buckets
    exec_heavy = int(np.sum(scores < execution_threshold))
    arch_heavy = int(np.sum(scores > architecture_threshold))
    balanced = total - exec_heavy - arch_heavy

    # Histogram
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    hist_counts, _ = np.histogram(scores, bins=bin_edges)

    # By seniority
    seniority_scores: dict[str, list[float]] = defaultdict(list)
    for j in jobs:
        seniority_scores[j.seniority].append(j.arch_exec_score)
    by_seniority = {
        level: round(float(np.mean(s)), 3)
        for level, s in seniority_scores.items()
    }

    # By company (top N by posting count)
    company_scores: dict[str, list[float]] = defaultdict(list)
    for j in jobs:
        company_scores[j.company].append(j.arch_exec_score)

    # Sort companies by posting count, take top N
    sorted_companies = sorted(
        company_scores.items(), key=lambda x: len(x[1]), reverse=True,
    )[:top_companies_count]
    by_company = {
        company: round(float(np.mean(s)), 3)
        for company, s in sorted_companies
    }

    # Top 5 most architectural postings
    sorted_by_arch = sorted(jobs, key=lambda j: j.arch_exec_score, reverse=True)
    top_arch = [
        {"title": j.title, "company": j.company, "score": j.arch_exec_score}
        for j in sorted_by_arch[:5]
    ]

    # Top 5 most execution-heavy postings
    sorted_by_exec = sorted(jobs, key=lambda j: j.arch_exec_score)
    top_exec = [
        {"title": j.title, "company": j.company, "score": j.arch_exec_score}
        for j in sorted_by_exec[:5]
    ]

    # Skill correlation by seniority
    skill_correlation: dict[str, float] = {}
    for level, level_scores in seniority_scores.items():
        level_jobs = [j for j in jobs if j.seniority == level]
        if len(level_jobs) > 2:
            arch_scores = [j.arch_exec_score for j in level_jobs]
            skill_counts = [j.skill_count for j in level_jobs]
            if np.std(arch_scores) > 0 and np.std(skill_counts) > 0:
                corr = float(np.corrcoef(arch_scores, skill_counts)[0, 1])
                skill_correlation[level] = round(corr, 3)

    result = ArchExecAnalytics(
        total_scored=total,
        mean_score=round(mean_score, 3),
        median_score=round(median_score, 3),
        std_score=round(std_score, 3),
        execution_heavy_count=exec_heavy,
        execution_heavy_pct=round(exec_heavy / total * 100, 1),
        architecture_heavy_count=arch_heavy,
        architecture_heavy_pct=round(arch_heavy / total * 100, 1),
        balanced_count=balanced,
        by_seniority=by_seniority,
        by_company=by_company,
        histogram_bins=bin_edges.tolist(),
        histogram_counts=hist_counts.tolist(),
        top_architectural_postings=top_arch,
        top_execution_postings=top_exec,
        skill_correlation=skill_correlation,
    )

    logger.info(
        "Arch-exec analytics: mean=%.2f, exec-heavy=%d (%.1f%%), arch-heavy=%d (%.1f%%)",
        mean_score, exec_heavy, exec_heavy / total * 100,
        arch_heavy, arch_heavy / total * 100,
    )
    return result
