"""
OpenAI embedding client with batching, rate limiting, and cost tracking.

Embeds ``Chunk`` objects via OpenAI's ``text-embedding-3-small``.
Every batch records a ``CostEntry`` on a shared ``CostLedger``; if
the ledger crosses the configured budget the run stops gracefully
and the remaining chunks keep ``embedding=None``.

Design choices worth flagging:

  - **Lazy client init**. ``OPENAI_API_KEY`` is checked only when an
    embed method is called, not at ``__init__``. This keeps the
    chunker + vector-store testable without an API key and without
    burning budget.
  - **Pluggable client factory**. Tests can inject a fake by passing
    ``client_factory=lambda key: MyFake()`` — useful for the smoke
    tests below and for any future integration tests.
  - **Retry policy**. Rate-limit / connection errors are retried up
    to three times with exponential backoff. Other OpenAI errors
    raise ``EmbedderError`` immediately so failures surface fast.

Usage::

    from src.embeddings.embedder import Embedder
    embedder = Embedder()
    chunks = embedder.embed_chunks(chunks)
    query_vec = embedder.embed_query("What skills are in demand?")
"""

import os
import time
from typing import Any, Callable

import yaml

from src.models import Chunk, CostEntry, CostLedger
from src.utils.io import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger("embeddings.Embedder")


class EmbedderError(Exception):
    """Raised when the embedder hits an unrecoverable error."""

    def __init__(self, message: str, original: Exception | None = None) -> None:
        self.original = original
        super().__init__(message)


class Embedder:
    """OpenAI embedding client with cost-tracked batching."""

    def __init__(
        self,
        cost_ledger: CostLedger | None = None,
        client_factory: Callable[[str], Any] | None = None,
    ) -> None:
        """
        Load config and prepare the embedder. The OpenAI client is
        NOT instantiated until the first embed call — see
        ``_ensure_client``.

        Args:
            cost_ledger: Optional shared CostLedger. A new one is
                created from settings.yaml when omitted.
            client_factory: Optional callable ``(api_key) -> client``
                used for dependency injection in tests. The factory
                result must expose ``embeddings.create(model, input,
                dimensions)`` with the same shape as OpenAI's SDK.
        """
        config_path = PROJECT_ROOT / "config" / "settings.yaml"
        with open(config_path, "r", encoding="utf-8") as fh:
            settings = yaml.safe_load(fh)

        oai = settings.get("openai", {})
        self._model: str = oai.get("embedding_model", "text-embedding-3-small")
        self._dimensions: int = oai.get("embedding_dimensions", 1536)
        self._batch_size: int = oai.get("embedding_batch_size", 50)
        self._rate_limit: float = oai.get("embedding_rate_limit_seconds", 0.3)
        self._cost_per_1k: float = oai.get("cost_per_1k_embedding_tokens", 0.00002)
        self._usd_to_eur: float = oai.get("usd_to_eur", 0.92)

        if cost_ledger is not None:
            self._ledger = cost_ledger
        else:
            budget = oai.get("budget_eur", 30.0)
            self._ledger = CostLedger(
                budget_eur=budget, usd_to_eur=self._usd_to_eur,
            )

        self._client_factory = client_factory
        self._client: Any | None = None

        logger.info(
            "Embedder initialized: model=%s, dims=%d, batch=%d",
            self._model, self._dimensions, self._batch_size,
        )

    @property
    def cost_ledger(self) -> CostLedger:
        """The CostLedger that this embedder writes to."""
        return self._ledger

    def _ensure_client(self) -> Any:
        """Build the OpenAI client on first use; cache it afterward."""
        if self._client is not None:
            return self._client

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key and self._client_factory is None:
            raise EmbedderError(
                "OPENAI_API_KEY is not set. Either export it or pass a "
                "client_factory to Embedder() for offline / testing use."
            )

        if self._client_factory is not None:
            self._client = self._client_factory(api_key)
            return self._client

        try:
            import openai
        except ImportError as exc:
            raise EmbedderError(
                "openai package not installed. Run: pip install openai", exc,
            )
        self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Embed all chunks in batches and write embeddings back in place.

        Stops early if the cost ledger crosses the budget; chunks
        after that point keep ``embedding=None`` so the caller can
        decide whether to drop them or resume later with a fresh
        budget.
        """
        if not chunks:
            return chunks

        total = len(chunks)
        embedded = 0
        logger.info(
            "Embedding %d chunks in batches of %d", total, self._batch_size,
        )

        for batch_num, start in enumerate(range(0, total, self._batch_size)):
            if self._ledger.is_over_budget():
                logger.warning(
                    "BUDGET EXCEEDED (%.2f / %.2f EUR). "
                    "Stopping embedding at chunk %d / %d.",
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
            except Exception as exc:
                logger.error("Batch %d failed: %s", batch_num, exc)
                continue

            for i, emb in enumerate(embeddings):
                chunks[start + i].embedding = emb
            embedded += len(embeddings)

            if end < total:
                time.sleep(self._rate_limit)

            if (batch_num + 1) % 10 == 0:
                logger.info(
                    "Progress: %d / %d chunks embedded (%.1f%%)",
                    embedded, total, embedded / total * 100,
                )

        logger.info(
            "Embedding complete: %d / %d chunks embedded. "
            "Total cost: $%.4f (%.4f EUR)",
            embedded, total,
            self._ledger.total_cost_usd,
            self._ledger.total_cost_eur,
        )
        return chunks

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query for vector search."""
        client = self._ensure_client()
        try:
            response = client.embeddings.create(
                model=self._model,
                input=[text],
                dimensions=self._dimensions,
            )
        except Exception as exc:
            raise EmbedderError(f"Query embedding failed: {exc}", original=exc)

        tokens = self._extract_tokens(response)
        if tokens:
            self._record_cost(tokens, batch_number=0, items=1, op="embed_query")

        return self._extract_vector(response, 0)

    def _embed_batch(
        self, texts: list[str], batch_number: int,
    ) -> list[list[float]]:
        """Call OpenAI for one batch, retrying transient failures."""
        client = self._ensure_client()
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(
                    model=self._model,
                    input=texts,
                    dimensions=self._dimensions,
                )
                tokens = self._extract_tokens(response)
                self._record_cost(tokens, batch_number, len(texts))
                return [
                    self._extract_vector(response, i) for i in range(len(texts))
                ]
            except Exception as exc:
                if self._is_transient(exc) and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Batch %d attempt %d/%d failed (%s). Retrying in %ds...",
                        batch_number, attempt + 1, max_retries, exc, wait,
                    )
                    time.sleep(wait)
                    continue
                raise EmbedderError(
                    f"Batch {batch_number} failed: {exc}", original=exc,
                )

        raise EmbedderError(
            f"Batch {batch_number} failed after {max_retries} retries"
        )

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        """Match OpenAI rate-limit / connection errors by class name."""
        try:
            import openai
        except ImportError:
            return False
        return isinstance(
            exc, (openai.RateLimitError, openai.APIConnectionError),
        )

    @staticmethod
    def _extract_vector(response: Any, index: int) -> list[float]:
        """Extract one embedding vector from an SDK or dict response."""
        if hasattr(response, "data"):
            item = response.data[index]
            return item.embedding if hasattr(item, "embedding") else item["embedding"]
        return response["data"][index]["embedding"]

    @staticmethod
    def _extract_tokens(response: Any) -> int:
        """Pull total_tokens off an SDK or dict response (0 if absent)."""
        usage = getattr(response, "usage", None)
        if usage is not None:
            return getattr(usage, "total_tokens", 0)
        return response.get("usage", {}).get("total_tokens", 0) if isinstance(response, dict) else 0

    def _record_cost(
        self,
        tokens_used: int,
        batch_number: int,
        items: int,
        op: str = "embed_batch",
    ) -> None:
        """Record one CostEntry on the ledger."""
        cost_usd = tokens_used * self._cost_per_1k / 1000
        cost_eur = cost_usd * self._usd_to_eur
        entry = CostEntry(
            operation=op,
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
