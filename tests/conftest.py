"""
Shared pytest fixtures.

The factories here build synthetic ``RawJob`` / ``EnrichedJob``
records covering the signals the pipeline cares about: AI roles in
high-risk domains with and without governance keywords, near-
duplicates across sources, intern + German signals, architecture-
heavy seniors, and so on. Tests compose these into small datasets
rather than hand-rolling Pydantic models inline.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# Make the project root importable for the test runner regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import EnrichedJob, EUAIActAnalysis, RawJob  # noqa: E402


_DIM = 1536


def _fake_embed(text: str) -> list[float]:
    """Token-bag deterministic embedding for offline tests."""
    vec = np.zeros(_DIM, dtype=np.float32)
    for token in text.lower().split():
        idx = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % _DIM
        vec[idx] += 1.0
    norm = float(np.linalg.norm(vec))
    return (vec / norm).tolist() if norm > 0 else vec.tolist()


@pytest.fixture
def fake_embed():
    """Expose the token-bag embedder to tests."""
    return _fake_embed


@pytest.fixture
def fake_openai_client():
    """
    Build a fake OpenAI client exposing ``embeddings.create``.

    Returned object matches the SDK shape closely enough for the
    Embedder to operate on. Each call returns one SimpleNamespace
    per input string with ``.embedding``, plus a ``.usage.total_tokens``.
    """
    class _FakeEmbeddings:
        def create(self, model, input, dimensions):
            vectors = [
                SimpleNamespace(embedding=_fake_embed(t)) for t in input
            ]
            usage = SimpleNamespace(
                total_tokens=sum(max(1, len(t.split())) for t in input)
            )
            return SimpleNamespace(data=vectors, usage=usage)

    class _FakeClient:
        def __init__(self):
            self.embeddings = _FakeEmbeddings()

    return _FakeClient()


@pytest.fixture
def fake_chat_client():
    """
    Fake chat client whose response echoes how many chunks it saw.

    Lets tests assert that the prompt builder fed the model what
    they expect without depending on the actual model output.
    """
    class _FakeChatCompletions:
        def create(self, model, messages, temperature, max_tokens):
            user = messages[-1]["content"]
            n_jobs = user.count("[Job ")
            n_arts = user.count("[Article ")
            answer = (
                f"Based on {n_jobs} job posting(s) and {n_arts} EU AI Act "
                "article(s), Python and SQL are the most-requested skills. "
                "Citing companies Acme Bank and Deepset, and Article 14 for "
                "human oversight."
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=answer)
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=500, completion_tokens=120,
                ),
            )

    class _FakeChat:
        def __init__(self):
            self.completions = _FakeChatCompletions()

    class _FakeClient:
        def __init__(self):
            self.chat = _FakeChat()

    return _FakeClient()


def _eu(is_ai=False, gap=False, domains=None, gov_kw=None, articles=None):
    """Compact factory for EUAIActAnalysis."""
    gov_kw = gov_kw or []
    return EUAIActAnalysis(
        is_ai_role=is_ai,
        touches_high_risk_domain=bool(domains),
        high_risk_domains=domains or [],
        annex_iii_sections=[],
        governance_keywords_found=gov_kw,
        governance_keyword_count=len(gov_kw),
        governance_gap=gap,
        relevant_articles=articles or [],
    )


def _enriched(
    company: str,
    title: str,
    description: str,
    source_id: str,
    *,
    source: str = "jsearch",
    score: float = 0.5,
    seniority: str = "mid",
    skills: list[str] | None = None,
    ai: bool = False,
    gap: bool = False,
    domains: list[str] | None = None,
    gov_kw: list[str] | None = None,
    date: str = "2026-05-01",
    hn_month: str | None = None,
    remote: bool = False,
    german: bool = False,
) -> EnrichedJob:
    skills = skills or []
    return EnrichedJob(
        company=company,
        title=title,
        location="Berlin, Germany",
        description=description,
        url=f"https://example.com/{source_id}",
        source=source,
        source_id=source_id,
        date_posted=date,
        hn_month=hn_month,
        skills={"C": skills},
        all_skills_flat=skills,
        skill_count=len(skills),
        arch_exec_score=score,
        arch_signals_found=[],
        exec_signals_found=[],
        seniority=seniority,
        requires_german=german,
        is_remote=remote,
        eu_ai_act=_eu(is_ai=ai, gap=gap, domains=domains, gov_kw=gov_kw,
                      articles=[9, 14] if domains else []),
    )


@pytest.fixture
def make_enriched():
    """Factory that tests can call to build EnrichedJob records."""
    return _enriched


@pytest.fixture
def make_raw():
    """Factory that tests can call to build RawJob records."""

    def factory(
        company: str, title: str, description: str, source_id: str,
        *, source: str = "jsearch", date: str = "2026-05-01",
        location: str = "Berlin, Germany",
    ) -> RawJob:
        return RawJob(
            company=company, title=title, location=location,
            description=description, date_posted=date,
            url=f"https://example.com/{source_id}",
            source=source, source_id=source_id,
        )

    return factory


@pytest.fixture
def sample_enriched_dataset(make_enriched) -> list[EnrichedJob]:
    """
    Twelve enriched jobs covering every dimension exercised by the
    analytics + RAG layers. Mirrors data/demo/sample_jobs.json in
    spirit but with explicit scores / domains for deterministic
    assertions.
    """
    return [
        make_enriched(
            "Acme Bank", "Senior ML Engineer, Credit Scoring",
            "Credit scoring and loan approval at Acme. " * 30,
            "a1", score=0.85, seniority="senior",
            skills=["python", "pytorch", "sql", "docker"],
            ai=True, gap=True, domains=["Essential_Services"],
        ),
        make_enriched(
            "Acme Bank", "ML Engineer, Fraud",
            "Fraud scoring and risk-based pricing. " * 30,
            "a2", score=0.45, seniority="mid",
            skills=["python", "sql"],
            ai=True, gap=True, domains=["Essential_Services"],
        ),
        make_enriched(
            "Acme Bank", "Junior Data Scientist",
            "Junior on the credit team. " * 30,
            "a3", score=0.15, seniority="junior",
            skills=["python", "sql"],
            ai=True, gap=True, domains=["Essential_Services"],
        ),
        make_enriched(
            "Deepset", "Senior RAG Engineer",
            "Build RAG systems with LangChain. " * 30,
            "d1", source="hackernews", score=0.80, seniority="senior",
            skills=["python", "rag", "langchain", "embeddings"],
            ai=True, hn_month="2026-04",
        ),
        make_enriched(
            "Deepset", "ML Engineer, GenAI",
            "Vector stores, embeddings. " * 30,
            "d2", source="hackernews", score=0.55, seniority="mid",
            skills=["python", "rag", "langchain"],
            ai=True, hn_month="2026-05",
        ),
        make_enriched(
            "Bio Health", "Data Scientist, Hiring Tools",
            "Resume screening with model bias audit and human oversight. " * 30,
            "b1", source="arbeitnow", score=0.40, seniority="mid",
            skills=["python", "scikit_learn"],
            ai=True, domains=["Employment"],
            gov_kw=["bias audit", "human oversight"],
        ),
        make_enriched(
            "Zalando", "Principal Engineer",
            "Lead the technical vision. " * 30,
            "z1", score=0.92, seniority="lead",
            skills=["python", "java"],
        ),
        make_enriched(
            "Zalando", "Data Engineer",
            "Write SQL queries, maintain pipelines. " * 30,
            "z2", score=0.20, seniority="mid",
            skills=["python", "sql"],
        ),
        make_enriched(
            "N26", "Werkstudent Data Analyst",
            "Werkstudentenstelle, Dashboards, Deutsch erforderlich. " * 20,
            "n1", source="bsj", score=0.10, seniority="intern",
            skills=["sql"], german=True,
        ),
        make_enriched(
            "Flixbus", "Junior Frontend Dev",
            "React. " * 60,
            "f1", source="arbeitnow", score=0.25, seniority="junior",
            skills=["typescript", "javascript"], remote=True,
        ),
        make_enriched(
            "TalentBridge", "Lead Engineer, Hiring Platform",
            "Applicant tracking. " * 30,
            "t1", source="hackernews", score=0.75, seniority="lead",
            skills=["python", "aws"],
            ai=True, gap=True, domains=["Employment"],
            hn_month="2026-03",
        ),
        make_enriched(
            "MediScan", "AI Engineer, Medical Imaging",
            "Train deep learning models. " * 30,
            "m1", source="arbeitnow", score=0.35, seniority="mid",
            skills=["python", "pytorch"],
            ai=True, gap=True, domains=["Healthcare"],
        ),
    ]
