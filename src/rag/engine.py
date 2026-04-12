"""
RAG query engine for Berlin AI Talent Radar.

Orchestrates retrieval + generation: classifies query intent, retrieves
relevant chunks from the vector store, builds a prompt, calls GPT-4o-mini,
scores confidence, and returns a RAGResult.

Usage::

    from src.rag.engine import RAGEngine
    engine = RAGEngine(vector_store, embedder)
    result = engine.query("What skills are most in demand?")
"""

import os
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
import yaml

from src.models import (
    CostEntry, CostLedger, RAGResult, SearchResult, ConfidenceLevel,
)
from src.embeddings.embedder import Embedder
from src.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from src.storage.numpy_store import NumpyVectorStore
from src.utils.Io import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger("rag.RAGEngine")


class QueryType(str, Enum):
    """Classification of user query intent."""
    LEGAL = "legal"
    MARKET = "market"
    MIXED = "mixed"


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for Berlin AI Talent Radar.

    Combines vector search over job postings and EU AI Act articles
    with GPT-4o-mini generation and multi-signal confidence scoring.
    """

    def __init__(
        self,
        vector_store: NumpyVectorStore,
        embedder: Embedder,
        cost_ledger: CostLedger | None = None,
    ) -> None:
        """
        Initialize RAG engine with dependencies.

        Args:
            vector_store: Loaded vector store for retrieval.
            embedder: Embedder for query embedding.
            cost_ledger: Shared cost ledger (uses embedder's if None).
        """
        self._store = vector_store
        self._embedder = embedder
        self._ledger = cost_ledger or embedder.cost_ledger

        # Load config
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh)

        rag = settings.get("rag", {})
        self._job_top_k = rag.get("default_job_top_k", 6)
        self._reg_top_k = rag.get("default_regulation_top_k", 3)
        self._legal_boost_reg_k = rag.get("legal_boost_regulation_top_k", 5)
        self._market_boost_job_k = rag.get("market_boost_job_top_k", 7)
        self._max_chunks = rag.get("max_total_chunks", 8)
        self._legal_keywords = set(rag.get("legal_keywords", []))
        self._market_keywords = set(rag.get("market_keywords", []))

        oai = settings.get("openai", {})
        self._chat_model = oai.get("chat_model", "gpt-4o-mini")
        self._chat_temp = oai.get("chat_temperature", 0.3)
        self._chat_max_tokens = oai.get("chat_max_tokens", 2000)
        self._cost_input = oai.get("cost_per_1k_input_tokens", 0.00015)
        self._cost_output = oai.get("cost_per_1k_output_tokens", 0.0006)
        self._usd_to_eur = oai.get("usd_to_eur", 0.92)

        rel = settings.get("reliability", {})
        self._weights = rel.get("weights", {})
        self._thresholds = rel.get("thresholds", {})
        self._freshness_days = rel.get("freshness_days", {})

        # OpenAI client for chat
        try:
            import openai
            self._client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            logger.error("openai package not installed")
            self._client = None

        logger.info(
            "RAGEngine initialized: model=%s, max_chunks=%d",
            self._chat_model, self._max_chunks,
        )

    def query(self, question: str) -> RAGResult:
        """
        Full RAG pipeline: classify -> retrieve -> generate -> score.

        Args:
            question: Natural language query from the user.

        Returns:
            RAGResult with answer, confidence scores, and citations.
        """
        logger.info("RAG query: %s", question[:100])

        # Step 1: Classify query
        query_type = self._classify_query(question)
        logger.info("Query classified as: %s", query_type.value)

        # Step 2: Embed query
        query_embedding = self._embedder.embed_query(question)

        # Step 3: Retrieve relevant chunks
        job_results, reg_results = self._retrieve(query_embedding, query_type)
        total_chunks = len(job_results) + len(reg_results)
        logger.info(
            "Retrieved %d job chunks + %d regulation chunks",
            len(job_results), len(reg_results),
        )

        # Step 4: Build prompt and generate
        messages = self._build_prompt(question, job_results, reg_results, query_type)
        answer, input_tokens, output_tokens = self._generate(messages)

        # Step 5: Record generation cost
        self._record_generation_cost(input_tokens, output_tokens)

        # Step 6: Score confidence
        confidence, scores = self._score_confidence(
            job_results, reg_results, answer,
        )

        # Step 7: Extract sources
        sources_jobs, sources_legal = self._extract_sources(
            job_results, reg_results,
        )

        result = RAGResult(
            answer=answer,
            confidence=confidence,
            confidence_scores=scores,
            sources_jobs=sources_jobs,
            sources_legal=sources_legal,
            num_chunks_used=total_chunks,
            query=question,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "RAG result: confidence=%s, chunks=%d, sources=%d jobs + %d legal",
            confidence, total_chunks, len(sources_jobs), len(sources_legal),
        )
        return result

    def _classify_query(self, question: str) -> QueryType:
        """Classify query as legal, market, or mixed."""
        q_lower = question.lower()
        legal_hits = sum(1 for kw in self._legal_keywords if kw in q_lower)
        market_hits = sum(1 for kw in self._market_keywords if kw in q_lower)

        if legal_hits > 0 and market_hits == 0:
            return QueryType.LEGAL
        if market_hits > 0 and legal_hits == 0:
            return QueryType.MARKET
        return QueryType.MIXED

    def _retrieve(
        self,
        query_embedding: list[float],
        query_type: QueryType,
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        """
        Retrieve relevant chunks, adjusting top_k by query type.

        Returns:
            Tuple of (job_results, regulation_results).
        """
        # Determine top_k per collection
        if query_type == QueryType.LEGAL:
            job_k = self._job_top_k
            reg_k = self._legal_boost_reg_k
        elif query_type == QueryType.MARKET:
            job_k = self._market_boost_job_k
            reg_k = self._reg_top_k
        else:
            job_k = self._job_top_k
            reg_k = self._reg_top_k

        # Retrieve from each collection
        job_results = self._store.search(
            query_embedding, source_type="job_posting", top_k=job_k,
        )
        reg_results = self._store.search(
            query_embedding, source_type="eu_ai_act", top_k=reg_k,
        )

        # Enforce max total chunks
        total = len(job_results) + len(reg_results)
        if total > self._max_chunks:
            # Merge, sort by similarity, take top max_chunks
            all_results = job_results + reg_results
            all_results.sort(key=lambda r: r.similarity, reverse=True)
            all_results = all_results[: self._max_chunks]
            job_results = [
                r for r in all_results
                if r.chunk.metadata.source_type == "job_posting"
            ]
            reg_results = [
                r for r in all_results
                if r.chunk.metadata.source_type == "eu_ai_act"
            ]

        return job_results, reg_results

    def _build_prompt(
        self,
        question: str,
        job_results: list[SearchResult],
        reg_results: list[SearchResult],
        query_type: QueryType,
    ) -> list[dict[str, str]]:
        """Build chat messages array for GPT-4o-mini."""
        user_content = build_user_prompt(
            question, job_results, reg_results, query_type.value,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _generate(
        self, messages: list[dict[str, str]],
    ) -> tuple[str, int, int]:
        """
        Call GPT-4o-mini and return the answer with token counts.

        Returns:
            Tuple of (answer_text, input_tokens, output_tokens).
        """
        if self._client is None:
            return (
                "Error: OpenAI client not available. Install openai package.",
                0, 0,
            )

        import openai

        try:
            response = self._client.chat.completions.create(
                model=self._chat_model,
                messages=messages,
                temperature=self._chat_temp,
                max_tokens=self._chat_max_tokens,
            )
            answer = response.choices[0].message.content or ""
            usage = response.usage
            return answer, usage.prompt_tokens, usage.completion_tokens

        except openai.OpenAIError as exc:
            logger.error("Generation failed: %s", exc)
            return f"Error generating answer: {exc}", 0, 0

    def _score_confidence(
        self,
        job_results: list[SearchResult],
        reg_results: list[SearchResult],
        answer: str,
    ) -> tuple[ConfidenceLevel, dict[str, float]]:
        """
        Compute multi-signal reliability score.

        Returns:
            Tuple of (confidence_level, scores_dict).
        """
        all_results = job_results + reg_results

        if not all_results:
            return "LOW", {"overall": 0.0}

        similarities = [r.similarity for r in all_results]

        # Signal 1: Consensus — average pairwise similarity among top chunks
        if len(similarities) > 1:
            consensus = float(np.mean(similarities))
        else:
            consensus = similarities[0] if similarities else 0.0

        # Signal 2: Coverage — fraction of chunks with decent similarity
        high_sim_count = sum(1 for s in similarities if s > 0.5)
        coverage = high_sim_count / len(similarities) if similarities else 0.0

        # Signal 3: Source diversity — distinct companies + articles
        companies = set()
        articles = set()
        for r in all_results:
            if r.chunk.metadata.source_type == "job_posting" and r.chunk.metadata.company:
                companies.add(r.chunk.metadata.company)
            elif r.chunk.metadata.source_type == "eu_ai_act" and r.chunk.metadata.article_number:
                articles.add(r.chunk.metadata.article_number)
        total_sources = len(companies) + len(articles)
        source_diversity = min(1.0, total_sources / max(len(all_results), 1))

        # Signal 4: Freshness — how recent are job postings
        freshness = self._compute_freshness(job_results)

        # Signal 5: Similarity distribution — tight clustering is good
        if len(similarities) > 1:
            std = float(np.std(similarities))
            # Lower std = more consistent = higher score
            sim_distribution = max(0.0, 1.0 - std * 3)
        else:
            sim_distribution = 0.5

        # Weighted average
        w = self._weights
        overall = (
            w.get("consensus", 0.25) * consensus
            + w.get("coverage", 0.30) * coverage
            + w.get("source_diversity", 0.15) * source_diversity
            + w.get("freshness", 0.15) * freshness
            + w.get("similarity_distribution", 0.15) * sim_distribution
        )

        # Map to confidence level
        high_threshold = self._thresholds.get("high", 0.70)
        medium_threshold = self._thresholds.get("medium", 0.45)

        if overall >= high_threshold:
            level: ConfidenceLevel = "HIGH"
        elif overall >= medium_threshold:
            level = "MEDIUM"
        else:
            level = "LOW"

        scores = {
            "consensus": round(consensus, 3),
            "coverage": round(coverage, 3),
            "source_diversity": round(source_diversity, 3),
            "freshness": round(freshness, 3),
            "similarity_distribution": round(sim_distribution, 3),
            "overall": round(overall, 3),
        }

        return level, scores

    def _compute_freshness(self, job_results: list[SearchResult]) -> float:
        """Compute freshness score based on posting dates."""
        if not job_results:
            return 0.5  # Neutral if no job data

        full_days = self._freshness_days.get("full", 30)
        half_days = self._freshness_days.get("half", 90)
        now = datetime.now(timezone.utc)

        scores: list[float] = []
        for r in job_results:
            date_str = r.chunk.metadata.date_posted
            if not date_str:
                scores.append(0.5)
                continue
            try:
                # Parse ISO date
                posted = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                if posted.tzinfo is None:
                    posted = posted.replace(tzinfo=timezone.utc)
                days_old = (now - posted).days
                if days_old <= full_days:
                    scores.append(1.0)
                elif days_old <= half_days:
                    scores.append(0.5)
                else:
                    scores.append(0.0)
            except (ValueError, TypeError):
                scores.append(0.5)

        return float(np.mean(scores)) if scores else 0.5

    def _extract_sources(
        self,
        job_results: list[SearchResult],
        reg_results: list[SearchResult],
    ) -> tuple[list[str], list[str]]:
        """Extract unique company names and article references."""
        companies: list[str] = []
        seen_companies: set[str] = set()
        for r in job_results:
            company = r.chunk.metadata.company
            if company and company not in seen_companies:
                companies.append(company)
                seen_companies.add(company)

        articles: list[str] = []
        seen_articles: set[int] = set()
        for r in reg_results:
            num = r.chunk.metadata.article_number
            if num and num not in seen_articles:
                articles.append(f"Article {num}")
                seen_articles.add(num)

        return companies, articles

    def _record_generation_cost(
        self, input_tokens: int, output_tokens: int,
    ) -> None:
        """Record chat generation cost on the CostLedger."""
        if input_tokens == 0 and output_tokens == 0:
            return

        cost_usd = (
            input_tokens * self._cost_input / 1000
            + output_tokens * self._cost_output / 1000
        )
        cost_eur = cost_usd * self._usd_to_eur

        entry = CostEntry(
            operation="generate",
            model=self._chat_model,
            tokens_used=input_tokens + output_tokens,
            cost_usd=cost_usd,
            cost_eur=cost_eur,
            items_processed=1,
        )
        self._ledger.entries.append(entry)

        logger.debug(
            "Generation cost: %d+%d tokens, $%.6f (%.6f EUR)",
            input_tokens, output_tokens, cost_usd, cost_eur,
        )
