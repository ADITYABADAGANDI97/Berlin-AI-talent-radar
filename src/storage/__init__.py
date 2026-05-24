"""
Storage package — Stage 4 vector persistence.

The default backend is ``NumpyVectorStore`` (cosine search over a
pre-normalized float32 matrix on disk). A pgvector backend is
configured in settings.yaml but not yet implemented.
"""

from src.storage.numpy_store import NumpyVectorStore

__all__ = ["NumpyVectorStore"]
