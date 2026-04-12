"""
Stage 4b: Vector Storage.

Provides numpy-backed vector store for cosine similarity search.

Usage::

    from src.storage import NumpyVectorStore
    store = NumpyVectorStore()
    store.save_job_chunks(chunks)
    store.load()
    results = store.search(query_embedding, top_k=6)
"""

from src.storage.numpy_store import NumpyVectorStore

__all__ = ["NumpyVectorStore"]
