"""
RAG query engine for Berlin AI Talent Radar.

Pipeline per query:

  1. Classify the question as legal / market / mixed by counting
     keyword hits (config-driven from settings.yaml).
  2. Embed the query via the Embedder.
  3. Retrieve job + regulation chunks with type-aware top_k (legal
     queries pull more regulations, market queries pull more jobs).
  4. Format a prompt with structured citation headers and ask
     GPT-4o-mini for an answer.
  5. Score confidence across five signals (consensus, coverage,
     source diversity, freshness, similarity distribution) and map
     to HIGH / MEDIUM / LOW via the reliability thresholds.
  6. Collect citations and return a fully-typed ``RAGResult``.

Like the Embedder, the chat client is built lazily so unit tests can
inject a fake via ``chat_client_factory`` without an API key.

Usage::

    from src.rag.engine import RAGEngine
    engine = RAGEngine(vector_store, embedder)
    result = engine.query("What skills are most in demand?")
"""

import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import numpy as np
import yaml

from src.embeddings.embedder import Embedder
from src.models import (
    ConfidenceLevel,
    CostEntry,
    CostLedger,
    RAGResult,
    SearchResult,
)
from src.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from src.storage.numpy_store import NumpyVectorStore
from src.utils.io import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger("rag.RAGEngine")


class QueryType(str, Enum):
    """Classification of user query intent."""

    LEGAL = "legal"
    MARKET = "market"
    MIXED = "mixed"


class RAGEngine:
    """Retrieval-Augmented Generation engine."""

    def __init__(
        self,
        vector_store: NumpyVectorStore,
        embedder: Embedder,
        cost_ledger: CostLedger | None = None,
        chat_client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        """
        Wire up the engine.

        Args:
            vector_store: A *loaded* NumpyVectorStore.
            embedder: Used to embed the query text.
            cost_ledger: Optional shared ledger; defaults to the one
                the embedder already writes to so all costs roll up.
            chat_client_factory: Optional ``(api_key) -> client`` for
                dependency injection (tests). The factory result must
                expose ``chat.completions.create(...)`` with the same
                shape as the OpenAI SDK.
        """
        self._store = vector_store
        self._embedder = embedder
        self._ledger = cost_ledger or embedder.cost_ledger

        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh)

        rag = settings.get("rag", {})
        self._job_top_k: int = rag.get("default_job_top_k", 6)
        self._reg_top_k: int = rag.get("default_regulation_top_k", 3)
        self._legal_boost_reg_k: int = rag.get("legal_boost_regulation_top_k", 5)
        self._market_boost_job_k: int = rag.get("market_boost_job_top_k", 7)
        self._max_chunks: int = rag.get("max_total_chunks", 8)
        self._legal_keywords: set[str] = set(rag.get("legal_keywords", []))
        self._market_keywords: set[str] = set(rag.get("market_keywords", []))

        oai = settings.get("openai", {})
        self._chat_model: str = oai.get("chat_model", "gpt-4o-mini")
        self._chat_temp: float = oai.get("chat_temperature", 0.3)
        self._chat_max_tokens: int = oai.get("chat_max_tokens", 2000)
        self._cost_input: float = oai.get("cost_per_1k_input_tokens", 0.00015)
        self._cost_output: float = oai.get("cost_per_1k_output_tokens", 0.0006)
        self._usd_to_eur: float = oai.get("usd_to_eur", 0.92)

        rel = settings.get("reliability", {})
        self._weights: dict[str, float] = rel.get("weights", {})
        self._thresholds: dict[str, float] = rel.get("thresholds", {})
        self._freshness_days: dict[str, int] = rel.get("freshness_days", {})

        self._chat_client_factory = chat_client_factory
        self._chat_client: Any | None = None

        logger.info(
            "RAGEngine initialized: model=%s, max_chunks=%d",
            self._chat_model, self._max_chunks,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str) -> RAGResult:
        """Run the full RAG pipeline and return a typed result."""
        logger.info("RAG query: %s", question[:100])

        query_type = self._classify_query(question)
        logger.info("Query classified as: %s", query_type.value)

        query_embedding = self._embedder.embed_query(question)

        job_results, reg_results = self._retrieve(query_embedding, query_type)
        total_chunks = len(job_results) + len(reg_results)
        logger.info(
            "Retrieved %d job chunks + %d regulation chunks",
            len(job_results), len(reg_results),
        )

        messages = self._build_prompt(
            question, job_results, reg_results, query_type,
        )
        answer, input_tokens, output_tokens = self._generate(messages)
        self._record_generation_cost(input_tokens, output_tokens)

        confidence, scores = self._score_confidence(job_results, reg_results)
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

    # ------------------------------------------------------------------
    # Query classification
    # ------------------------------------------------------------------

    def _classify_query(self, question: str) -> QueryType:
        q_lower = question.lower()
        legal_hits = sum(1 for kw in self._legal_keywords if kw in q_lower)
        market_hits = sum(1 for kw in self._market_keywords if kw in q_lower)
        if legal_hits > 0 and market_hits == 0:
            return QueryType.LEGAL
        if market_hits > 0 and legal_hits == 0:
            return QueryType.MARKET
        return QueryType.MIXED

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        query_embedding: list[float],
        query_type: QueryType,
    ) -> tuple[list[SearchResult], list[SearchResult]]:
        if query_type == QueryType.LEGAL:
            job_k, reg_k = self._job_top_k, self._legal_boost_reg_k
        elif query_type == QueryType.MARKET:
            job_k, reg_k = self._market_boost_job_k, self._reg_top_k
        else:
            job_k, reg_k = self._job_top_k, self._reg_top_k

        job_results = self._store.search(
            query_embedding, source_type="job_posting", top_k=job_k,
        )
        reg_results = self._store.search(
            query_embedding, source_type="eu_ai_act", top_k=reg_k,
        )

        # Hard cap on total chunks to keep prompts compact + costs bounded
        total = len(job_results) + len(reg_results)
        if total > self._max_chunks:
            merged = sorted(
                job_results + reg_results,
                key=lambda r: r.similarity, reverse=True,
            )[: self._max_chunks]
            job_results = [
                r for r in merged
                if r.chunk.metadata.source_type == "job_posting"
            ]
            reg_results = [
                r for r in merged
                if r.chunk.metadata.source_type == "eu_ai_act"
            ]
        return job_results, reg_results

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        question: str,
        job_results: list[SearchResult],
        reg_results: list[SearchResult],
        query_type: QueryType,
    ) -> list[dict[str, str]]:
        user_content = build_user_prompt(
            question, job_results, reg_results, query_type.value,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _ensure_chat_client(self) -> Any:
        if self._chat_client is not None:
            return self._chat_client

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if self._chat_client_factory is not None:
            self._chat_client = self._chat_client_factory(api_key)
            return self._chat_client

        if not api_key:
            return None

        try:
            import openai
        except ImportError:
            logger.error("openai package not installed")
            return None
        self._chat_client = openai.OpenAI(api_key=api_key)
        return self._chat_client

    def _generate(
        self, messages: list[dict[str, str]],
    ) -> tuple[str, int, int]:
        """Return ``(answer, prompt_tokens, completion_tokens)``."""
        client = self._ensure_chat_client()
        if client is None:
            return (
                "Error: OpenAI chat client is unavailable. "
                "Set OPENAI_API_KEY or supply a chat_client_factory.",
                0, 0,
            )
        try:
            response = client.chat.completions.create(
                model=self._chat_model,
                messages=messages,
                temperature=self._chat_temp,
                max_tokens=self._chat_max_tokens,
            )
        except Exception as exc:
            logger.error("Generation failed: %s", exc)
            return f"Error generating answer: {exc}", 0, 0

        answer = self._extract_answer(response)
        prompt_tokens, completion_tokens = self._extract_token_counts(response)
        return answer, prompt_tokens, completion_tokens

    @staticmethod
    def _extract_answer(response: Any) -> str:
        """Pull the assistant message off an SDK or dict response."""
        choices = getattr(response, "choices", None) or response.get("choices")
        first = choices[0]
        message = getattr(first, "message", None) or first.get("message", {})
        content = (
            getattr(message, "content", None)
            if not isinstance(message, dict)
            else message.get("content")
        )
        return content or ""

    @staticmethod
    def _extract_token_counts(response: Any) -> tuple[int, int]:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage", {})
        if usage is None:
            return 0, 0
        prompt = (
            getattr(usage, "prompt_tokens", None)
            if not isinstance(usage, dict) else usage.get("prompt_tokens")
        )
        completion = (
            getattr(usage, "completion_tokens", None)
            if not isinstance(usage, dict) else usage.get("completion_tokens")
        )
        return int(prompt or 0), int(completion or 0)

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def _score_confidence(
        self,
        job_results: list[SearchResult],
        reg_results: list[SearchResult],
    ) -> tuple[ConfidenceLevel, dict[str, float]]:
        """Combine five signals into one HIGH / MEDIUM / LOW verdict."""
        all_results = job_results + reg_results
        if not all_results:
            return "LOW", {"overall": 0.0}

        similarities = [r.similarity for r in all_results]

        # 1. Consensus — mean similarity across retrieved chunks
        consensus = float(np.mean(similarities))

        # 2. Coverage — fraction of chunks with similarity above 0.5
        coverage = sum(1 for s in similarities if s > 0.5) / len(similarities)

        # 3. Source diversity — distinct companies + articles vs chunk count
        companies: set[str] = set()
        articles: set[int] = set()
        for r in all_results:
            meta = r.chunk.metadata
            if meta.source_type == "job_posting" and meta.company:
                companies.add(meta.company)
            elif meta.source_type == "eu_ai_act" and meta.article_number:
                articles.add(meta.article_number)
        total_sources = len(companies) + len(articles)
        source_diversity = min(1.0, total_sources / max(len(all_results), 1))

        # 4. Freshness — recency of job postings (legal text always fresh)
        freshness = self._compute_freshness(job_results)

        # 5. Similarity distribution — tighter clustering => more agreement
        if len(similarities) > 1:
            std = float(np.std(similarities))
            sim_distribution = max(0.0, 1.0 - std * 3)
        else:
            sim_distribution = 0.5

        w = self._weights
        overall = (
            w.get("consensus", 0.25) * consensus
            + w.get("coverage", 0.30) * coverage
            + w.get("source_diversity", 0.15) * source_diversity
            + w.get("freshness", 0.15) * freshness
            + w.get("similarity_distribution", 0.15) * sim_distribution
        )

        high = self._thresholds.get("high", 0.70)
        medium = self._thresholds.get("medium", 0.45)
        if overall >= high:
            level: ConfidenceLevel = "HIGH"
        elif overall >= medium:
            level = "MEDIUM"
        else:
            level = "LOW"

        return level, {
            "consensus": round(consensus, 3),
            "coverage": round(coverage, 3),
            "source_diversity": round(source_diversity, 3),
            "freshness": round(freshness, 3),
            "similarity_distribution": round(sim_distribution, 3),
            "overall": round(overall, 3),
        }

    def _compute_freshness(self, job_results: list[SearchResult]) -> float:
        """Per-posting recency score averaged across job chunks."""
        if not job_results:
            return 0.5  # neutral when there's no job data

        full = self._freshness_days.get("full", 30)
        half = self._freshness_days.get("half", 90)
        now = datetime.now(timezone.utc)

        scores: list[float] = []
        for r in job_results:
            date_str = r.chunk.metadata.date_posted
            if not date_str:
                scores.append(0.5)
                continue
            try:
                posted = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                if posted.tzinfo is None:
                    posted = posted.replace(tzinfo=timezone.utc)
                days_old = (now - posted).days
                if days_old <= full:
                    scores.append(1.0)
                elif days_old <= half:
                    scores.append(0.5)
                else:
                    scores.append(0.0)
            except (ValueError, TypeError):
                scores.append(0.5)
        return float(np.mean(scores)) if scores else 0.5

    # ------------------------------------------------------------------
    # Citations
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sources(
        job_results: list[SearchResult],
        reg_results: list[SearchResult],
    ) -> tuple[list[str], list[str]]:
        companies: list[str] = []
        seen_c: set[str] = set()
        for r in job_results:
            c = r.chunk.metadata.company
            if c and c not in seen_c:
                companies.append(c)
                seen_c.add(c)

        articles: list[str] = []
        seen_a: set[int] = set()
        for r in reg_results:
            n = r.chunk.metadata.article_number
            if n and n not in seen_a:
                articles.append(f"Article {n}")
                seen_a.add(n)
        return companies, articles

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def _record_generation_cost(
        self, input_tokens: int, output_tokens: int,
    ) -> None:
        if input_tokens == 0 and output_tokens == 0:
            return
        cost_usd = (
            input_tokens * self._cost_input / 1000
            + output_tokens * self._cost_output / 1000
        )
        cost_eur = cost_usd * self._usd_to_eur
        self._ledger.entries.append(CostEntry(
            operation="generate",
            model=self._chat_model,
            tokens_used=input_tokens + output_tokens,
            cost_usd=cost_usd,
            cost_eur=cost_eur,
            items_processed=1,
        ))
        logger.debug(
            "Generation cost: %d+%d tokens, $%.6f (%.6f EUR)",
            input_tokens, output_tokens, cost_usd, cost_eur,
        )
