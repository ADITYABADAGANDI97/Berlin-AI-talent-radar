"""
Stage 4a: Chunking & Embedding.

Provides text chunking and OpenAI embedding with cost tracking.

Usage::

    from src.embeddings import Chunker, Embedder
    chunker = Chunker()
    embedder = Embedder()
    chunks = chunker.chunk_jobs(enriched_jobs)
    chunks = embedder.embed_chunks(chunks)
"""

from src.embeddings.chunker import Chunker
from src.embeddings.embedder import Embedder, EmbedderError

__all__ = ["Chunker", "Embedder", "EmbedderError"]
