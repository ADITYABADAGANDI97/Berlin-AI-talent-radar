"""
Stage 4c: Retrieval-Augmented Generation.

Provides the RAG query engine with confidence scoring.

Usage::

    from src.rag import RAGEngine
    engine = RAGEngine(vector_store, embedder)
    result = engine.query("What skills are most in demand?")
"""

from src.rag.engine import RAGEngine, QueryType
from src.rag.prompts import SYSTEM_PROMPT, build_user_prompt

__all__ = ["RAGEngine", "QueryType", "SYSTEM_PROMPT", "build_user_prompt"]
