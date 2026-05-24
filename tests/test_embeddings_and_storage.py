"""
Tests for src/embeddings/ + src/storage/.

Exercise the chunker on EnrichedJob, embed via a fake OpenAI client
so no network or API key is required, then round-trip the result
through the NumpyVectorStore and verify cosine ranking + metadata
filters.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def test_chunker_emits_chunks_with_full_metadata(make_enriched):
    from src.embeddings import Chunker

    # >400 words → chunker should split into multiple overlapping chunks
    job = make_enriched(
        "Deepset", "Senior RAG Engineer",
        "Build retrieval-augmented systems with LangChain. " * 150,
        "d1", score=0.78, skills=["python", "rag"],
        ai=True, domains=None,
    )
    chunks = Chunker().chunk_jobs([job])
    assert len(chunks) >= 2, "long description should split into multiple chunks"

    chunk0 = chunks[0]
    assert chunk0.metadata.source_type == "job_posting"
    assert chunk0.metadata.company == "Deepset"
    assert chunk0.metadata.arch_exec_score == 0.78
    assert chunk0.metadata.skills == ["python", "rag"]
    assert chunk0.chunk_index == 0
    assert chunk0.total_chunks == len(chunks)


def test_chunker_handles_eu_articles():
    from src.embeddings import Chunker

    articles = [{
        "article_number": 14,
        "article_title": "Human oversight",
        "enforcement_date": "2026-08-02",
        "penalty_reference": "Article 99",
        "text": "High-risk AI systems shall be effectively overseen by "
                "natural persons during the period in which the AI system "
                "is in use. " * 30,
    }]
    chunks = Chunker().chunk_eu_articles(articles)
    assert chunks
    assert chunks[0].metadata.source_type == "eu_ai_act"
    assert chunks[0].metadata.article_number == 14
    assert chunks[0].metadata.enforcement_date == "2026-08-02"


def test_embedder_lazy_init_raises_without_key(monkeypatch):
    from src.embeddings import Embedder, EmbedderError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embedder = Embedder()  # constructs fine
    with pytest.raises(EmbedderError, match="OPENAI_API_KEY"):
        embedder.embed_query("test")


def test_embedder_with_factory_works_offline(
    make_enriched, fake_openai_client,
):
    from src.embeddings import Chunker, Embedder

    chunks = Chunker().chunk_jobs([
        make_enriched(
            "Deepset", "Senior RAG",
            "Build RAG systems. " * 30, "d1",
            skills=["python", "rag"],
        )
    ])
    embedder = Embedder(client_factory=lambda _: fake_openai_client)
    out = embedder.embed_chunks(chunks)
    assert all(c.embedding is not None for c in out)
    assert all(len(c.embedding) == 1536 for c in out)
    assert embedder.cost_ledger.entries, "should record one cost entry"


def test_numpy_store_roundtrip_and_search(
    tmp_path, make_enriched, fake_openai_client, fake_embed,
):
    from src.embeddings import Chunker, Embedder
    from src.storage import NumpyVectorStore

    jobs = [
        make_enriched(
            "Acme Bank", "Senior ML Engineer, Credit Scoring",
            "ML engineer for credit scoring at Acme Bank. " * 25,
            "a1", score=0.5, skills=["python", "sql"],
            ai=True, gap=True, domains=["Essential_Services"],
        ),
        make_enriched(
            "Deepset", "Senior RAG Engineer",
            "Senior RAG engineer building retrieval systems. " * 25,
            "d1", score=0.8, skills=["python", "rag", "langchain"],
            ai=True,
        ),
    ]

    chunks = Chunker().chunk_jobs(jobs)
    embedder = Embedder(client_factory=lambda _: fake_openai_client)
    chunks = embedder.embed_chunks(chunks)

    store = NumpyVectorStore()
    store._jobs_npz = tmp_path / "job_chunks.npz"
    store._regs_npz = tmp_path / "regulation_chunks.npz"
    store.save_job_chunks(chunks)

    # Reload from disk
    store2 = NumpyVectorStore()
    store2._jobs_npz = store._jobs_npz
    store2._regs_npz = store._regs_npz
    store2.load()
    assert store2.job_chunk_count == len(chunks)

    # Ranking — RAG query should put Deepset on top
    rag_results = store2.search(
        fake_embed("senior RAG retrieval engineer"), top_k=3,
    )
    assert rag_results[0].chunk.metadata.company == "Deepset"

    # Filter — governance_gap=True returns only Acme Bank
    gap_results = store2.search(
        fake_embed("engineer"), top_k=10,
        filters={"governance_gap": True},
    )
    assert gap_results
    assert all(r.chunk.metadata.company == "Acme Bank" for r in gap_results)

    # Filter — skills list-membership
    skill_results = store2.search(
        fake_embed("engineer"), top_k=10, filters={"skills": "rag"},
    )
    assert skill_results
    assert all(r.chunk.metadata.company == "Deepset" for r in skill_results)
