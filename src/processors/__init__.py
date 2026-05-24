"""
Processors package — Stage 3 enrichment pipeline.

Transforms ``RawJob`` (from collectors) into ``EnrichedJob`` with
skills, arch-exec score, EU AI Act analysis, and seniority metadata.

The orchestrator is ``EnrichmentPipeline``; the individual processors
are exposed for unit-testing and ad-hoc reuse.
"""

from src.processors.arch_exec_scorer import ArchExecScorer
from src.processors.base import BaseProcessor, ProcessorError
from src.processors.cleaner import Cleaner
from src.processors.deduplicator import Deduplicator
from src.processors.governance_analyzer import GovernanceAnalyzer
from src.processors.metadata_detector import MetadataDetector
from src.processors.pipeline import EnrichmentPipeline
from src.processors.skill_extractor import SkillExtractor

__all__ = [
    "BaseProcessor",
    "ProcessorError",
    "Cleaner",
    "SkillExtractor",
    "ArchExecScorer",
    "GovernanceAnalyzer",
    "MetadataDetector",
    "Deduplicator",
    "EnrichmentPipeline",
]
