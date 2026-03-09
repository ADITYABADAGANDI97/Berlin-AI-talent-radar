"""
src/models.py — Berlin AI Talent Radar: Core Data Models
=========================================================
Schema-first design: every pipeline stage operates on typed Pydantic models.
No loose dictionaries. All models are immutable-by-convention and validated
at construction time via Pydantic v2.

Model hierarchy:
    RawJob              → raw collector output
    EUAIActAnalysis     → governance analysis result (embedded in EnrichedJob)
    EnrichedJob         → post-processing output (extends RawJob)
    ChunkMetadata       → metadata attached to every vector chunk
    Chunk               → embeddable text unit with full provenance
    SearchResult        → retrieval output (Chunk + similarity score)
    RAGResult           → full RAG answer with confidence and citations
    CostEntry           → single API cost event
    CostLedger          → running cost tracker
    AnalyticsResult     → analytics engine output (passed to dashboard + report)
    PipelineStatus      → status command output

Usage:
    from src.models import RawJob, EnrichedJob, RAGResult
    job = RawJob(company="Deepset", title="RAG Engineer", ...)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source literals
# ---------------------------------------------------------------------------
SourceType = Literal["jsearch", "hackernews", "arbeitnow", "bsj", "demo"]
ChunkSourceType = Literal["job_posting", "eu_ai_act"]
SeniorityLevel = Literal["intern", "junior", "mid", "senior", "lead"]
ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]
VectorBackend = Literal["numpy", "pgvector"]


# ===========================================================================
# Stage 1: Raw Collection
# ===========================================================================

class RawJob(BaseModel):
    """
    Raw job posting as returned by a collector, before any enrichment.

    Fields mirror the minimum common schema across all data sources.
    ``source_id`` is unique per source — it is the primary deduplication key
    within a single source. Cross-source deduplication is handled by the
    cleaner using fuzzy matching on (company, title).

    Example::

        job = RawJob(
            company="Deepset",
            title="RAG Engineer",
            location="Berlin, Germany",
            description="We are looking for...",
            url="https://...",
            source="jsearch",
            source_id="jsearch_abc123",
        )
    """

    company: str = Field(..., description="Employer name as listed in the posting")
    title: str = Field(..., description="Job title as listed in the posting")
    location: str = Field(..., description="Location string from the posting")
    description: str = Field(..., description="Full job description text (may contain HTML)")
    date_posted: str | None = Field(default=None, description="ISO date string or relative label")
    url: str = Field(..., description="Canonical URL for the posting")
    source: SourceType = Field(..., description="Data source identifier")
    source_id: str = Field(..., description="Unique ID within source — used for idempotent re-runs")
    hn_month: str | None = Field(
        default=None,
        description="HN Who's Hiring month tag, e.g. '2025-08'. Only populated for HN source. "
                    "Enables 6-month skill trend analysis.",
    )

    @field_validator("company", "title", "location", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        """Strip leading/trailing whitespace from string fields."""
        return v.strip() if isinstance(v, str) else v

    @field_validator("description", mode="before")
    @classmethod
    def require_non_empty_description(cls, v: str) -> str:
        """Ensure description is not empty after stripping."""
        stripped = v.strip() if isinstance(v, str) else ""
        if not stripped:
            raise ValueError("Job description must not be empty")
        return stripped


# ===========================================================================
# Stage 2: EU AI Act Analysis
# ===========================================================================

class EUAIActAnalysis(BaseModel):
    """
    Result of EU AI Act governance analysis for a single job posting.

    The governance_gap field is the core signal:
        governance_gap = is_ai_role AND touches_high_risk_domain AND
                         governance_keyword_count == 0

    A governance gap means the company is building/deploying AI in a regulated
    domain (Annex III) without any apparent awareness of their compliance
    obligations under Articles 9-15 and 26.

    Example::

        analysis = EUAIActAnalysis(
            is_ai_role=True,
            touches_high_risk_domain=True,
            high_risk_domains=["Employment"],
            annex_iii_sections=["Section 4"],
            governance_keywords_found=[],
            governance_keyword_count=0,
            governance_gap=True,
            relevant_articles=[9, 13, 14, 26, 50, 99],
        )
    """

    is_ai_role: bool = Field(
        ...,
        description="True if the posting is for an AI/ML/data science role",
    )
    touches_high_risk_domain: bool = Field(
        ...,
        description="True if the role involves an Annex III high-risk AI domain",
    )
    high_risk_domains: list[str] = Field(
        default_factory=list,
        description="Matched Annex III domains, e.g. ['Employment', 'Healthcare']",
    )
    annex_iii_sections: list[str] = Field(
        default_factory=list,
        description="Annex III section labels, e.g. ['Section 4: Employment']",
    )
    governance_keywords_found: list[str] = Field(
        default_factory=list,
        description="Governance/compliance keywords found in the posting",
    )
    governance_keyword_count: int = Field(
        default=0,
        description="Number of distinct governance keywords found",
        ge=0,
    )
    governance_gap: bool = Field(
        ...,
        description="True = AI role in regulated domain with ZERO governance mentions",
    )
    relevant_articles: list[int] = Field(
        default_factory=list,
        description="EU AI Act article numbers applicable to this posting",
    )

    @model_validator(mode="after")
    def validate_governance_gap_logic(self) -> "EUAIActAnalysis":
        """
        Governance gap is only possible when all three conditions are true.
        Validate logical consistency of the computed fields.
        """
        if self.governance_gap:
            if not self.is_ai_role:
                raise ValueError("governance_gap=True requires is_ai_role=True")
            if not self.touches_high_risk_domain:
                raise ValueError("governance_gap=True requires touches_high_risk_domain=True")
        return self


# ===========================================================================
# Stage 3: Enriched Job (post-processing)
# ===========================================================================

class EnrichedJob(RawJob):
    """
    Job posting after full processing pipeline: cleaning → skill extraction →
    governance analysis → arch-exec scoring → seniority detection.

    Extends RawJob with all derived signals. The description field at this
    stage contains cleaned text (HTML stripped, normalized whitespace).

    Note on skills dict: keys are category names from skill_taxonomy.yaml,
    values are lists of matched skill canonical names.

    Example::

        job = EnrichedJob(
            ...base RawJob fields...,
            skills={"LLM & GenAI": ["rag", "langchain"], "MLOps": ["docker"]},
            all_skills_flat=["rag", "langchain", "docker"],
            skill_count=3,
            arch_exec_score=0.72,
            arch_signals_found=["system design", "roadmap"],
            exec_signals_found=["maintain pipelines"],
            seniority="senior",
            requires_german=False,
            is_remote=True,
            eu_ai_act=EUAIActAnalysis(...),
        )
    """

    # Skill extraction outputs
    skills: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Skills by category: {'LLM & GenAI': ['rag', 'langchain']}",
    )
    all_skills_flat: list[str] = Field(
        default_factory=list,
        description="Flattened list of all matched skill canonical names",
    )
    skill_count: int = Field(
        default=0,
        description="Total number of distinct skills matched",
        ge=0,
    )

    # Architecture-Execution Spectrum
    arch_exec_score: float = Field(
        default=0.5,
        description="0.0=pure execution → 1.0=pure architecture",
        ge=0.0,
        le=1.0,
    )
    arch_signals_found: list[str] = Field(
        default_factory=list,
        description="Architecture signals detected in the description",
    )
    exec_signals_found: list[str] = Field(
        default_factory=list,
        description="Execution signals detected in the description",
    )
    arch_raw_score: float = Field(
        default=0.0,
        description="Raw weighted architecture signal count (before normalization)",
        ge=0.0,
    )
    exec_raw_score: float = Field(
        default=0.0,
        description="Raw weighted execution signal count (before normalization)",
        ge=0.0,
    )

    # Role classification
    seniority: SeniorityLevel = Field(
        default="mid",
        description="Seniority level derived from title and description keywords",
    )
    requires_german: bool = Field(
        default=False,
        description="True if German language is required (not just preferred)",
    )
    is_remote: bool = Field(
        default=False,
        description="True if role is remote or hybrid",
    )

    # EU AI Act governance
    eu_ai_act: EUAIActAnalysis = Field(
        ...,
        description="Full EU AI Act governance analysis result",
    )

    # Processing metadata
    cleaned_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="UTC ISO timestamp of when this job was enriched",
    )

    @model_validator(mode="after")
    def sync_skill_count(self) -> "EnrichedJob":
        """Ensure skill_count matches len(all_skills_flat) if not explicitly set."""
        if self.skill_count == 0 and self.all_skills_flat:
            object.__setattr__(self, "skill_count", len(self.all_skills_flat))
        return self


# ===========================================================================
# Stage 4: Chunking & Embedding
# ===========================================================================

class ChunkMetadata(BaseModel):
    """
    Metadata attached to every vector chunk. Enables filtered search and
    full provenance reconstruction from retrieved chunks.

    Job posting fields are None for EU AI Act chunks; legal fields are None
    for job posting chunks. source_type is the discriminant.

    Example (job chunk)::

        meta = ChunkMetadata(
            source_type="job_posting",
            company="N26",
            title="ML Engineer, Fraud Detection",
            location="Berlin, Germany",
            source="jsearch",
            url="https://...",
            skills=["python", "pytorch", "docker"],
            arch_exec_score=0.45,
            seniority="mid",
            is_high_risk=True,
            high_risk_domains=["Essential_Services"],
            governance_gap=True,
            date_posted="2025-11-01",
        )

    Example (EU AI Act chunk)::

        meta = ChunkMetadata(
            source_type="eu_ai_act",
            article_number=14,
            article_title="Human oversight",
            enforcement_date="2026-08-02",
            penalty_reference="Article 99",
        )
    """

    source_type: ChunkSourceType = Field(
        ...,
        description="'job_posting' | 'eu_ai_act' — used for filtered vector search",
    )

    # --- Job posting fields (None for EU AI Act chunks) ---
    company: str | None = Field(default=None, description="Employer name")
    title: str | None = Field(default=None, description="Job title")
    location: str | None = Field(default=None, description="Location string")
    source: SourceType | None = Field(default=None, description="Data source")
    url: str | None = Field(default=None, description="Posting URL")
    skills: list[str] | None = Field(default=None, description="Flat list of matched skills")
    arch_exec_score: float | None = Field(default=None, description="Arch-exec spectrum score")
    seniority: SeniorityLevel | None = Field(default=None, description="Seniority level")
    is_high_risk: bool | None = Field(default=None, description="Touches high-risk AI domain")
    high_risk_domains: list[str] | None = Field(default=None, description="Matched Annex III domains")
    governance_gap: bool | None = Field(default=None, description="Has governance gap flag")
    date_posted: str | None = Field(default=None, description="Posting date")
    hn_month: str | None = Field(default=None, description="HN month tag for trend analysis")

    # --- EU AI Act fields (None for job chunks) ---
    article_number: int | None = Field(default=None, description="EU AI Act article number")
    article_title: str | None = Field(default=None, description="Article title")
    enforcement_date: str | None = Field(default=None, description="Enforcement date: 2026-08-02")
    penalty_reference: str | None = Field(default=None, description="Penalty context string")


class Chunk(BaseModel):
    """
    A single embeddable text unit with full metadata provenance.

    The text field contains the raw (cleaned) text slice. The embedding
    field is None until the embed stage runs; it is populated in-place by
    the embedder and then stored in the vector store.

    chunk_index tracks position within the parent document for reconstruction.

    Example::

        chunk = Chunk(
            text="We are looking for a senior ML engineer...",
            chunk_index=0,
            total_chunks=3,
            metadata=ChunkMetadata(source_type="job_posting", ...),
        )
    """

    text: str = Field(..., description="The text content of this chunk")
    embedding: list[float] | None = Field(
        default=None,
        description="1536-dim embedding vector (text-embedding-3-small). None until embedded.",
    )
    metadata: ChunkMetadata = Field(..., description="Full provenance metadata")
    chunk_index: int = Field(default=0, description="Position within parent document", ge=0)
    total_chunks: int = Field(default=1, description="Total chunks from parent document", ge=1)

    @field_validator("text", mode="before")
    @classmethod
    def require_non_empty_text(cls, v: str) -> str:
        """Chunk text must not be empty."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Chunk text must not be empty")
        return v


# ===========================================================================
# Stage 5: Retrieval & RAG
# ===========================================================================

class SearchResult(BaseModel):
    """
    A single result from vector similarity search.

    Returned by VectorStore.search(). The similarity score is cosine similarity,
    ranging from 0.0 (orthogonal) to 1.0 (identical).

    Example::

        result = SearchResult(chunk=chunk, similarity=0.87)
    """

    chunk: Chunk = Field(..., description="The retrieved chunk with full metadata")
    similarity: float = Field(
        ...,
        description="Cosine similarity score [0.0, 1.0]",
        ge=0.0,
        le=1.0,
    )


class RAGResult(BaseModel):
    """
    Complete output from the RAG engine for a single query.

    Includes the generated answer, confidence assessment, and full citation
    provenance for both job data and legal sources.

    Example::

        result = RAGResult(
            answer="Based on 47 job postings, Python is required in 89% of roles...",
            confidence="HIGH",
            confidence_scores={
                "consensus": 0.82,
                "coverage": 0.91,
                "source_diversity": 0.75,
                "freshness": 0.95,
                "similarity_distribution": 0.68,
                "overall": 0.84,
            },
            sources_jobs=["Deepset", "N26", "Delivery Hero"],
            sources_legal=["Article 14", "Article 26"],
            num_chunks_used=9,
            query="What skills are most in demand for RAG engineers?",
        )
    """

    answer: str = Field(..., description="Generated answer text (Big 4 consulting quality)")
    confidence: ConfidenceLevel = Field(
        ...,
        description="HIGH (≥0.70) | MEDIUM (≥0.45) | LOW (<0.45)",
    )
    confidence_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-signal and overall confidence scores",
    )
    sources_jobs: list[str] = Field(
        default_factory=list,
        description="Company names cited from job data",
    )
    sources_legal: list[str] = Field(
        default_factory=list,
        description="EU AI Act article references cited, e.g. 'Article 14'",
    )
    num_chunks_used: int = Field(
        default=0,
        description="Total chunks passed to the generator",
        ge=0,
    )
    query: str = Field(default="", description="Original query string for display")
    generated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="UTC ISO timestamp of generation",
    )


# ===========================================================================
# Cost Tracking
# ===========================================================================

class CostEntry(BaseModel):
    """
    A single API cost event for cost ledger tracking.

    Example::

        entry = CostEntry(
            operation="embed_batch",
            model="text-embedding-3-small",
            tokens_used=12500,
            cost_usd=0.00025,
            cost_eur=0.00023,
            batch_number=3,
            items_processed=50,
        )
    """

    operation: str = Field(..., description="Operation type: 'embed_batch' | 'generate'")
    model: str = Field(..., description="OpenAI model used")
    tokens_used: int = Field(..., description="Total tokens consumed", ge=0)
    cost_usd: float = Field(..., description="Cost in USD", ge=0.0)
    cost_eur: float = Field(..., description="Cost in EUR (converted)", ge=0.0)
    batch_number: int = Field(default=0, description="Batch number for embed operations", ge=0)
    items_processed: int = Field(default=0, description="Chunks or queries processed", ge=0)
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="UTC ISO timestamp",
    )


class CostLedger(BaseModel):
    """
    Running cost tracker for the full pipeline run.

    Tracks cumulative spend against the project budget with a warning
    threshold to prevent overruns.

    Example::

        ledger = CostLedger(budget_eur=30.0, warning_threshold_pct=0.80)
        ledger.entries.append(entry)
        assert ledger.total_cost_eur < ledger.budget_eur
    """

    budget_eur: float = Field(default=30.0, description="Total project budget in EUR", gt=0)
    warning_threshold_pct: float = Field(
        default=0.80,
        description="Log a WARNING when spend exceeds this fraction of budget",
        ge=0.0,
        le=1.0,
    )
    entries: list[CostEntry] = Field(default_factory=list, description="All cost events")
    usd_to_eur: float = Field(default=0.92, description="USD→EUR conversion rate", gt=0)

    @property
    def total_cost_usd(self) -> float:
        """Total spend in USD across all entries."""
        return sum(e.cost_usd for e in self.entries)

    @property
    def total_cost_eur(self) -> float:
        """Total spend in EUR across all entries."""
        return sum(e.cost_eur for e in self.entries)

    @property
    def budget_remaining_eur(self) -> float:
        """Remaining budget in EUR."""
        return self.budget_eur - self.total_cost_eur

    @property
    def budget_used_pct(self) -> float:
        """Fraction of budget consumed [0.0, 1.0+]."""
        return self.total_cost_eur / self.budget_eur

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed across all entries."""
        return sum(e.tokens_used for e in self.entries)

    def is_over_budget(self) -> bool:
        """Return True if total spend exceeds budget."""
        return self.total_cost_eur >= self.budget_eur

    def is_near_budget(self) -> bool:
        """Return True if spend has passed the warning threshold."""
        return self.budget_used_pct >= self.warning_threshold_pct


# ===========================================================================
# Analytics Results
# ===========================================================================

class SkillAnalytics(BaseModel):
    """
    Output of skill_analytics.py — passed to dashboard Tab 2 and report Section 1.

    All counts are absolute; percentages are relative to total_jobs.

    Example::

        analytics = SkillAnalytics(
            total_jobs=750,
            skill_counts={"python": 600, "docker": 450, ...},
            ...
        )
    """

    total_jobs: int = Field(..., description="Total enriched jobs analyzed", ge=0)
    skill_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Skill canonical name → absolute posting count",
    )
    skill_percentages: dict[str, float] = Field(
        default_factory=dict,
        description="Skill canonical name → percentage of total_jobs",
    )
    category_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Category name → total skill mentions in that category",
    )
    top_20_skills: list[tuple[str, int]] = Field(
        default_factory=list,
        description="Ordered list of (skill_name, count) for top 20 skills",
    )
    monthly_trends: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="HN-sourced trends: skill → {month: count}. e.g. {'rag': {'2025-08': 3}}",
    )
    skill_growth: dict[str, str] = Field(
        default_factory=dict,
        description="skill → 'explosive' | 'steady' | 'declining'",
    )
    genai_count: int = Field(default=0, description="Postings requiring any GenAI/LLM skill")
    traditional_ml_count: int = Field(default=0, description="Postings requiring traditional ML only")
    docker_count: int = Field(default=0, description="Postings mentioning Docker")
    phd_count: int = Field(default=0, description="Postings requiring PhD")
    masters_count: int = Field(default=0, description="Postings requiring Masters")
    german_required_count: int = Field(default=0, description="Postings requiring German")
    remote_count: int = Field(default=0, description="Remote/hybrid postings")
    german_pct: float = Field(default=0.0)
    remote_pct: float = Field(default=0.0)
    genai_pct: float = Field(default=0.0)
    traditional_ml_pct: float = Field(default=0.0)


class GovernanceAnalytics(BaseModel):
    """
    Output of governance_analytics.py — passed to dashboard Tab 4 and report Section 3.

    Example::

        analytics = GovernanceAnalytics(
            total_ai_roles=620,
            high_risk_count=187,
            high_risk_pct=30.2,
            governance_mention_count=12,
            governance_gap_count=175,
            governance_gap_pct=93.6,
            ...
        )
    """

    total_ai_roles: int = Field(..., description="Total postings classified as AI roles", ge=0)
    high_risk_count: int = Field(..., description="AI roles touching Annex III domains", ge=0)
    high_risk_pct: float = Field(..., description="high_risk_count / total_ai_roles * 100")
    governance_mention_count: int = Field(..., description="High-risk roles with ≥1 governance keyword", ge=0)
    governance_gap_count: int = Field(..., description="High-risk AI roles with ZERO governance keywords", ge=0)
    governance_gap_pct: float = Field(..., description="governance_gap_count / high_risk_count * 100")
    by_domain: dict[str, int] = Field(
        default_factory=dict,
        description="Annex III domain → count of high-risk postings",
    )
    by_company: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Company → {ai_roles, high_risk, governance_mentions, has_gap}",
    )
    article_coverage: dict[int, dict[str, Any]] = Field(
        default_factory=dict,
        description="Article number → {title, postings_mentioning, pct}",
    )
    enforcement_date: str = Field(default="2026-08-02")
    days_to_enforcement: int = Field(default=0, description="Days remaining until enforcement", ge=0)
    max_penalty_eur: int = Field(default=35_000_000)


class ArchExecAnalytics(BaseModel):
    """
    Output of arch_exec_analytics.py — passed to dashboard Tab 3 and report Section 2.

    Example::

        analytics = ArchExecAnalytics(
            total_scored=750,
            execution_heavy_count=300,
            execution_heavy_pct=40.0,
            architecture_heavy_count=187,
            architecture_heavy_pct=24.9,
            ...
        )
    """

    total_scored: int = Field(..., description="Total postings with arch-exec scores", ge=0)
    mean_score: float = Field(..., description="Mean arch-exec score across all postings")
    median_score: float = Field(..., description="Median arch-exec score")
    std_score: float = Field(..., description="Standard deviation of scores")
    execution_heavy_count: int = Field(..., description="Postings with score < 0.40")
    execution_heavy_pct: float = Field(...)
    architecture_heavy_count: int = Field(..., description="Postings with score > 0.70")
    architecture_heavy_pct: float = Field(...)
    balanced_count: int = Field(..., description="Postings with 0.40 ≤ score ≤ 0.70")
    by_seniority: dict[str, float] = Field(
        default_factory=dict,
        description="Seniority level → mean arch-exec score",
    )
    by_company: dict[str, float] = Field(
        default_factory=dict,
        description="Company → mean arch-exec score (top 15 companies)",
    )
    histogram_bins: list[float] = Field(
        default_factory=list,
        description="Bin edges for 10-bucket histogram [0.0, 0.1, ..., 1.0]",
    )
    histogram_counts: list[int] = Field(
        default_factory=list,
        description="Posting count per histogram bin",
    )
    top_architectural_postings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top 5 most architectural postings with title/company/score",
    )
    top_execution_postings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top 5 most execution-heavy postings with title/company/score",
    )
    skill_correlation: dict[str, float] = Field(
        default_factory=dict,
        description="Correlation between arch_exec_score and skill_count by seniority",
    )


class CompanyAnalytics(BaseModel):
    """
    Output of company_analytics.py — passed to dashboard Tab 1 and report Section 4.

    Example::

        analytics = CompanyAnalytics(
            total_companies=87,
            rankings=[{"company": "Zalando", "count": 23, ...}],
            ...
        )
    """

    total_companies: int = Field(..., description="Unique companies in dataset", ge=0)
    rankings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Companies sorted by posting count with metadata",
    )
    skill_profiles: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Company → top 8 skills",
    )
    avg_arch_exec: dict[str, float] = Field(
        default_factory=dict,
        description="Company → average arch-exec score",
    )
    governance_gaps: dict[str, bool] = Field(
        default_factory=dict,
        description="Company → True if any postings have governance_gap",
    )
    seniority_mix: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Company → {seniority_level: count}",
    )


class AnalyticsResult(BaseModel):
    """
    Aggregated output from the full analytics engine.

    This is the primary data contract between the analytics layer and both
    the Streamlit dashboard and the report generator.

    Example::

        result = AnalyticsResult(
            generated_at="2025-11-15T10:30:00",
            total_jobs=750,
            skills=SkillAnalytics(...),
            governance=GovernanceAnalytics(...),
            arch_exec=ArchExecAnalytics(...),
            companies=CompanyAnalytics(...),
        )
    """

    generated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="UTC ISO timestamp of analytics computation",
    )
    total_jobs: int = Field(..., description="Total enriched jobs in analysis", ge=0)
    total_chunks: int = Field(default=0, description="Total chunks in vector store")
    data_sources: list[str] = Field(default_factory=list, description="Sources contributing data")
    date_range: dict[str, str] = Field(
        default_factory=dict,
        description="{'earliest': 'YYYY-MM-DD', 'latest': 'YYYY-MM-DD'}",
    )
    skills: SkillAnalytics = Field(..., description="Skill landscape analytics")
    governance: GovernanceAnalytics = Field(..., description="EU AI Act governance analytics")
    arch_exec: ArchExecAnalytics = Field(..., description="Architecture-Execution analytics")
    companies: CompanyAnalytics = Field(..., description="Company intelligence analytics")
    cost_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline cost summary from CostLedger",
    )


# ===========================================================================
# Pipeline Status
# ===========================================================================

class PipelineStatus(BaseModel):
    """
    Output of ``python main.py status`` — human-readable pipeline state.

    Example::

        status = PipelineStatus(
            raw_jobs={"jsearch": 412, "hackernews": 187, "arbeitnow": 63, "bsj": 41},
            total_raw_jobs=703,
            total_enriched_jobs=698,
            total_chunks=4234,
            embeddings_ready=True,
            vector_backend="numpy",
            total_cost_eur=2.34,
            budget_remaining_eur=27.66,
        )
    """

    raw_jobs: dict[str, int] = Field(
        default_factory=dict,
        description="Source → raw job count",
    )
    total_raw_jobs: int = Field(default=0, ge=0)
    total_enriched_jobs: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=0, ge=0)
    embeddings_ready: bool = Field(default=False)
    vector_backend: VectorBackend = Field(default="numpy")
    total_cost_eur: float = Field(default=0.0, ge=0.0)
    budget_remaining_eur: float = Field(default=30.0)
    last_collect: str | None = Field(default=None, description="ISO timestamp of last collect run")
    last_process: str | None = Field(default=None, description="ISO timestamp of last process run")
    last_embed: str | None = Field(default=None, description="ISO timestamp of last embed run")
    last_analyze: str | None = Field(default=None, description="ISO timestamp of last analyze run")
    report_path: str | None = Field(default=None, description="Path to generated report")
    is_demo_mode: bool = Field(default=False, description="True if running with sample data")