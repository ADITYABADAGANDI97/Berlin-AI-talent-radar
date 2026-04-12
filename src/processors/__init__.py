"""
Stage 3: Enrichment Processors.

Transforms RawJob objects into fully-typed EnrichedJob objects via a
chain of five processors: cleaning, skill extraction, arch-exec scoring,
governance analysis, and metadata detection.

Usage::

    from src.processors import EnrichmentPipeline
    pipeline = EnrichmentPipeline()
    enriched_jobs = pipeline.enrich(raw_jobs)
"""

from src.processors.arch_exec_scorer import ArchExecScorer
from src.processors.base import BaseProcessor, ProcessorError
from src.processors.cleaner import Cleaner
from src.processors.governance_analyzer import GovernanceAnalyzer
from src.processors.metadata_detector import MetadataDetector
from src.processors.pipeline import EnrichmentPipeline
from src.processors.skill_extractor import SkillExtractor

__all__ = [
    "EnrichmentPipeline",
    "BaseProcessor",
    "ProcessorError",
    "Cleaner",
    "SkillExtractor",
    "ArchExecScorer",
    "GovernanceAnalyzer",
    "MetadataDetector",
]
