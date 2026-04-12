"""
OpenAI embedding client with batching, rate limiting, and cost tracking.

Embeds Chunk objects via the OpenAI text-embedding-3-small model. Tracks
all costs on a CostLedger and enforces budget limits — stops embedding
if the budget is exceeded mid-run.

Usage::

    from src.embeddings.embedder import Embedder
    embedder = Embedder()
    chunks = embedder.embed_chunks(chunks)
    query_vec = embedder.embed_query("What skills are in demand?")
"""

import os
import time
from typing import Any

import yaml

from src.models import CostEntry, CostLedger, Chunk
from src.utils.Io import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger("embeddings.Embedder")


class EmbedderError(Exception):
    """Raised when the embedder encounters an unrecoverable error."""

    def __init__(self, message: str, original: Exception | None = None) -> None:
        self.original = original
        super().__init__(message)


class Embedder:
    """
    OpenAI embedding client with batching, rate limiting, and budget
    enforcement via CostLedger.
    """

    def __init__(self, cost_ledger: CostLedger | None = None) -> None:
        """
        Initialize OpenAI client and load config.

        Args:
            cost_ledger: Optional shared CostLedger. Creates a new one
                from config if not provided.

        Raises:
            EmbedderError: If OPENAI_API_KEY is not set.
        """
        # Load config
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh)

        oai = settings.get("openai", {})
        self._model = oai.get("embedding_model", "text-embedding-3-small")
        self._dimensions = oai.get("embedding_dimensions", 1536)
        self._batch_size = oai.get("embedding_batch_size", 50)
        self._rate_limit = oai.get("embedding_rate_limit_seconds", 0.3)
        self._cost_per_1k = oai.get("cost_per_1k_embedding_tokens", 0.00002)
        self._usd_to_eur = oai.get("usd_to_eur", 0.92)

        # Cost ledger
        if cost_ledger is not None:
            self._ledger = cost_ledger
        else:
            budget = oai.get("budget_eur", 30.0)
            self._ledger = CostLedger(
                budget_eur=budget, usd_to_eur=self._usd_to_eur,
            )

        # OpenAI client (lazy import to avoid hard dependency at import time)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EmbedderError(
                "OPENAI_API_KEY environment variable is not set. "
                "Get your key from https://platform.openai.com/api-keys"
            )

        try:
            import openai
            self._client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise EmbedderError(
                "openai package not installed. Run: pip install openai"
            )

        logger.info(
            "Embedder initialized: model=%s, dims=%d, batch=%d",
            self._model, self._dimensions, self._batch_size,
        )

    @property
    def cost_ledger(self) -> CostLedger:
        """Return the current cost ledger."""
        return self._ledger

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Embed all chunks in batches, populating chunk.embedding.

        Stops early if the budget is exceeded. Remaining chunks keep
        embedding=None.

        Args:
            chunks: List of Chunk objects with embedding=None.

        Returns:
            Same list with embedding fields populated where possible.
        """
        total = len(chunks)
        embedded = 0
        logger.info("Embedding %d chunks in batches of %d", total, self._batch_size)

        for batch_num, start in enumerate(range(0, total, self._batch_size)):
            # Budget check
            if self._ledger.is_over_budget():
                logger.warning(
                    "BUDGET EXCEEDED (%.2f EUR / %.2f EUR). "
                    "Stopping embedding at chunk %d/%d.",
                    self._ledger.total_cost_eur,
                    self._ledger.budget_eur,
                    start, total,
                )
                break

            if self._ledger.is_near_budget():
                logger.warning(
                    "Budget %.0f%% used (%.2f / %.2f EUR)",
                    self._ledger.budget_used_pct * 100,
                    self._ledger.total_cost_eur,
                    self._ledger.budget_eur,
                )

            end = min(start + self._batch_size, total)
            batch_texts = [c.text for c in chunks[start:end]]

            try:
                embeddings = self._embed_batch(batch_texts, batch_num)
                for i, emb in enumerate(embeddings):
                    # Use object.__setattr__ since Pydantic models may be frozen
                    object.__setattr__(chunks[start + i], "embedding", emb)
                embedded += len(embeddings)
            except Exception as exc:
                logger.error("Batch %d failed: %s", batch_num, exc)
                # Continue with next batch rather than failing entirely
                continue

            # Rate limit between batches
            if end < total:
                time.sleep(self._rate_limit)

            # Progress logging
            if (batch_num + 1) % 10 == 0:
                logger.info(
                    "Progress: %d/%d chunks embedded (%.1f%%)",
                    embedded, total, embedded / total * 100,
                )

        logger.info(
            "Embedding complete: %d/%d chunks embedded. "
            "Total cost: $%.4f (%.4f EUR)",
            embedded, total,
            self._ledger.total_cost_usd,
            self._ledger.total_cost_eur,
        )
        return chunks

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string for vector search.

        Args:
            text: Query text to embed.

        Returns:
            1536-dim embedding vector.
        """
        import openai

        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=[text],
                dimensions=self._dimensions,
            )
            return response.data[0].embedding
        except openai.OpenAIError as exc:
            raise EmbedderError(f"Query embedding failed: {exc}", original=exc)

    def _embed_batch(
        self, texts: list[str], batch_number: int,
    ) -> list[list[float]]:
        """
        Call OpenAI embeddings API for a single batch.

        Retries up to 3 times with exponential backoff on rate limit
        and connection errors.

        Args:
            texts: List of text strings.
            batch_number: Batch index for logging.

        Returns:
            List of embedding vectors.
        """
        import openai

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                    dimensions=self._dimensions,
                )

                # Record cost
                tokens = response.usage.total_tokens
                self._record_cost(tokens, batch_number, len(texts))

                # Return embeddings in input order
                return [item.embedding for item in response.data]

            except (openai.RateLimitError, openai.APIConnectionError) as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Batch %d attempt %d/%d failed (%s). Retrying in %ds...",
                    batch_number, attempt + 1, max_retries, exc, wait,
                )
                time.sleep(wait)
            except openai.OpenAIError as exc:
                raise EmbedderError(
                    f"Batch {batch_number} failed: {exc}", original=exc,
                )

        raise EmbedderError(f"Batch {batch_number} failed after {max_retries} retries")

    def _record_cost(
        self, tokens_used: int, batch_number: int, items: int,
    ) -> None:
        """Record a CostEntry on the ledger after each batch."""
        cost_usd = tokens_used * self._cost_per_1k / 1000
        cost_eur = cost_usd * self._usd_to_eur

        entry = CostEntry(
            operation="embed_batch",
            model=self._model,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            cost_eur=cost_eur,
            batch_number=batch_number,
            items_processed=items,
        )
        self._ledger.entries.append(entry)

        logger.debug(
            "Batch %d: %d tokens, $%.6f (%.6f EUR). Running total: $%.4f",
            batch_number, tokens_used, cost_usd, cost_eur,
            self._ledger.total_cost_usd,
        )
