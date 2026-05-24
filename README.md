# Berlin AI Talent Radar

> Production-grade RAG intelligence platform analyzing Berlin's AI job market
> against EU AI Act compliance. Built by a Data Engineer in Berlin.

A typed, multi-stage pipeline that collects AI job postings from four
sources, scores each role on an original **Architecture-Execution Spectrum**,
checks every posting against the **EU AI Act** (Regulation EU 2024/1689,
enforceable 2 August 2026), and surfaces the results through both a
Streamlit dashboard and a natural-language RAG chat.

## What it does

1. **Collects** AI / ML / data postings from JSearch (LinkedIn + Indeed +
   Glassdoor), Hacker News "Who is Hiring?" (6-month historical), Arbeitnow,
   and Berlin Startup Jobs.
2. **Enriches** every posting with a 73-skill taxonomy match, an
   Architecture-Execution score (0.0 = pure execution, AI-vulnerable →
   1.0 = pure architecture, AI-amplified), seniority detection, German /
   remote signals, and a full EU AI Act analysis.
3. **Detects governance gaps**: AI roles in Annex III high-risk domains
   (employment, essential services, healthcare, education, …) where the
   posting mentions zero governance keywords (audit, oversight, bias,
   compliance, transparency, …).
4. **Embeds** chunks with `text-embedding-3-small`, stores them in a
   numpy-backed vector store with metadata filters.
5. **Answers questions** over the dataset with GPT-4o-mini, citing
   companies for market evidence and article numbers for legal evidence,
   and scoring confidence across five signals.
6. **Renders** a Streamlit dashboard and a markdown executive report.

## Quick start

No API keys required for the demo path:

```bash
git clone <repo-url>
cd berlin-ai-talent-radar
pip install -r requirements.txt

python main.py demo                # synthetic data → analytics + report
streamlit run app/app.py           # open the dashboard
```

The demo runs end-to-end on twelve hand-crafted sample postings
(`data/demo/sample_jobs.json`), produces `data/reports/analytics.json`
and `data/reports/berlin_ai_talent_radar_report.md`, and powers all five
dashboard tabs except RAG Chat (which needs the embed step).

## Full pipeline (with API keys)

```bash
cp .env.example .env               # add RAPIDAPI_KEY + OPENAI_API_KEY
source .env

python main.py collect             # Stage 2: pull jobs from all sources
python main.py process             # Stage 3: enrich raw → EnrichedJob
python main.py embed               # Stage 4: chunk + embed + persist
python main.py analyze             # Stage 6: build analytics.json
python main.py report              # Stage 8: render markdown report
python main.py query "What governance obligations apply to credit scoring under Article 14?"

# Or do all of the above in one go:
python main.py full

# Show what's been run so far:
python main.py status
```

The pipeline enforces a configurable EUR budget on the cost ledger
(`config/settings.yaml → openai.budget_eur`, default €30) and stops
embedding cleanly if the budget is exceeded mid-run.

## Architecture

```
config/                      # All tuneable knobs (YAML, no code changes)
  settings.yaml              #   master config
  skill_taxonomy.yaml        #   73 skills × 8 categories
  arch_exec_signals.yaml     #   Architecture-Execution Spectrum signals
  governance_taxonomy.yaml   #   Annex III domains + governance keywords
  eu_ai_act_articles.yaml    #   Article texts for RAG retrieval
  search_queries.yaml        #   JSearch queries — swap to retarget markets

src/
  models.py                  # Pydantic v2 models — schema-first design
  collectors/                # Stage 2: source-specific HTTP / scrape clients
  processors/                # Stage 3: clean → dedup → skills → scores → governance
  embeddings/                # Stage 4: chunk + embed
  storage/                   # Stage 4: numpy vector store with metadata filters
  rag/                       # Stage 5: classify → retrieve → generate → score confidence
  analytics/                 # Stage 6: four pure-function analytics modules + engine
  reports/                   # Stage 8: AnalyticsResult → markdown report

app/app.py                   # Streamlit dashboard (5 tabs)
main.py                      # CLI orchestrator
tests/                       # pytest suite — no network, runs in ~2s
```

Every stage is decoupled by a typed contract from `src/models.py`:
`RawJob → EnrichedJob → Chunk → SearchResult → RAGResult` and
`AnalyticsResult`. Swap a collector, processor, or storage backend
without touching the rest of the pipeline.

## Status

| Stage | Status |
|-------|--------|
| 1. Config + Pydantic models | ✅ |
| 2. Collectors (JSearch, HN, Arbeitnow, BSJ, EU AI Act) | ✅ |
| 3. Processors (clean, skills, arch-exec, governance, metadata, dedup) | ✅ |
| 4. Embeddings + numpy vector store | ✅ |
| 5. RAG engine + reliability scoring | ✅ |
| 6. Analytics (skills, governance, arch-exec, companies) | ✅ |
| 7. Streamlit dashboard + CLI | ✅ |
| 8. Markdown executive report | ✅ |
| Tests | ✅ — 19/19 passing |

## Tech stack

- **Python 3.10+**, **Pydantic v2** for typed models
- **OpenAI** `text-embedding-3-small` + `gpt-4o-mini` (lazy init, mockable)
- **numpy** for cosine search (pgvector backend configured but optional)
- **Streamlit + Plotly + pandas** for the dashboard
- **pytest** for the test suite (no network, runs offline)

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

The suite covers every layer — processors, embeddings, vector store,
RAG engine, analytics, report generator — by injecting fake OpenAI
clients via the `client_factory` hook. No API key, no network, no cost.

## Built by

Aditya Badagandi — M.Sc. Data Science, AI & Digital Business
(Gisma University of Applied Sciences)
