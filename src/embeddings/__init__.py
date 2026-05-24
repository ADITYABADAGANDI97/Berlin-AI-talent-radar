"""
Embeddings package — Stage 4.

Chunks enriched jobs and EU AI Act articles, then embeds them via
OpenAI's text-embedding-3-small. Vectors are persisted by the
``storage`` package.
"""

from src.embeddings.chunker import Chunker
from src.embeddings.embedder import Embedder, EmbedderError

__all__ = ["Chunker", "Embedder", "EmbedderError"]
