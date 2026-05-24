"""
Tests for src/rag/.

Use the fake_openai_client + fake_chat_client fixtures so we cover
classification, retrieval, prompt construction, generation, cost
recording, and citation extraction without any network call.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def loaded_store(
    tmp_path, make_enriched, fake_openai_client,
):
    """Build a NumpyVectorStore populated with two AI postings + two articles."""
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
            "RAG engineer building retrieval systems. " * 25,
            "d1", score=0.8, skills=["python", "rag"],
            ai=True,
        ),
    ]
    articles = [
        {
            "article_number": 14,
            "article_title": "Human oversight",
            "enforcement_date": "2026-08-02",
            "penalty_reference": "Article 99",
            "text": "High-risk AI systems shall be designed to be "
                    "overseen by natural persons during operation. " * 10,
        },
        {
            "article_number": 26,
            "article_title": "Obligations of deployers",
            "enforcement_date": "2026-08-02",
            "penalty_reference": "Article 99",
            "text": "Deployers of high-risk AI systems shall take "
                    "appropriate technical and organisational measures. " * 10,
        },
    ]

    chunker = Chunker()
    job_chunks = chunker.chunk_jobs(jobs)
    reg_chunks = chunker.chunk_eu_articles(articles)

    embedder = Embedder(client_factory=lambda _: fake_openai_client)
    job_chunks = embedder.embed_chunks(job_chunks)
    reg_chunks = embedder.embed_chunks(reg_chunks)

    store = NumpyVectorStore()
    store._jobs_npz = tmp_path / "job_chunks.npz"
    store._regs_npz = tmp_path / "regulation_chunks.npz"
    store.save_job_chunks(job_chunks)
    store.save_regulation_chunks(reg_chunks)

    store2 = NumpyVectorStore()
    store2._jobs_npz = store._jobs_npz
    store2._regs_npz = store._regs_npz
    store2.load()
    return store2, embedder


def test_query_classification(loaded_store, fake_chat_client):
    from src.rag import QueryType, RAGEngine

    store, embedder = loaded_store
    engine = RAGEngine(
        store, embedder, chat_client_factory=lambda _: fake_chat_client,
    )
    assert engine._classify_query(
        "What does Article 14 of the EU AI Act require?"
    ) == QueryType.LEGAL
    assert engine._classify_query(
        "What skills are most in demand for Berlin ML engineers?"
    ) == QueryType.MARKET
    assert engine._classify_query(
        "Which Berlin companies need governance skills for hiring AI?"
    ) == QueryType.MIXED


def test_end_to_end_query_returns_typed_result(loaded_store, fake_chat_client):
    from src.rag import RAGEngine

    store, embedder = loaded_store
    engine = RAGEngine(
        store, embedder, chat_client_factory=lambda _: fake_chat_client,
    )
    result = engine.query("What skills are most in demand in Berlin?")

    assert result.confidence in ("HIGH", "MEDIUM", "LOW")
    assert result.num_chunks_used > 0
    assert result.sources_jobs, "should cite at least one company"
    assert result.sources_legal, "should cite at least one article"
    assert any(s.startswith("Article ") for s in result.sources_legal)
    assert "Python" in result.answer or "SQL" in result.answer


def test_generation_cost_recorded(loaded_store, fake_chat_client):
    from src.rag import RAGEngine

    store, embedder = loaded_store
    engine = RAGEngine(
        store, embedder, chat_client_factory=lambda _: fake_chat_client,
    )

    pre = len([
        e for e in embedder.cost_ledger.entries if e.operation == "generate"
    ])
    engine.query("Quick sanity check question")
    post = len([
        e for e in embedder.cost_ledger.entries if e.operation == "generate"
    ])
    assert post == pre + 1
    last = [
        e for e in embedder.cost_ledger.entries if e.operation == "generate"
    ][-1]
    assert last.tokens_used == 620  # 500 prompt + 120 completion from fake


def test_engine_without_client_returns_helpful_error(
    loaded_store, monkeypatch,
):
    from src.rag import RAGEngine

    store, embedder = loaded_store
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    engine = RAGEngine(store, embedder)  # no factory, no key

    # embed_query needs a real client too, so we monkey-patch the embedder
    # to return a known vector and bypass the OpenAI call.
    embedder._client_factory = lambda _: type(
        "X", (), {"embeddings": type(
            "Y", (), {"create": lambda self, **kw: type(
                "Z", (), {"data": [type("D", (), {"embedding": [0.0] * 1536})()],
                          "usage": type("U", (), {"total_tokens": 1})()})()})()},
    )()

    result = engine.query("anything")
    assert "OPENAI_API_KEY" in result.answer or "chat_client_factory" in result.answer
