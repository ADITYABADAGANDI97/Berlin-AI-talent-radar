"""
RAG package — Stage 5.

Combines vector retrieval over job postings and EU AI Act articles
with GPT-4o-mini answer generation and a five-signal reliability
score (consensus / coverage / source diversity / freshness /
similarity distribution).
"""

from src.rag.engine import QueryType, RAGEngine
from src.rag.prompts import SYSTEM_PROMPT, build_user_prompt

__all__ = ["RAGEngine", "QueryType", "SYSTEM_PROMPT", "build_user_prompt"]
