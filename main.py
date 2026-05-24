"""
Berlin AI Talent Radar — CLI orchestrator.

Entry point for every pipeline stage. Run stages individually or
chain them with ``full``. ``demo`` runs end-to-end on synthetic
sample data and requires no API keys.

Usage::

    python main.py collect              # Stage 2: pull raw jobs from collectors
    python main.py process              # Stage 3: enrich raw -> EnrichedJob
    python main.py embed                # Stage 4: chunk + embed + persist
    python main.py analyze              # Stage 6: produce analytics.json
    python main.py query "..."          # Stage 5: ask the RAG engine
    python main.py status               # what's been run, what hasn't
    python main.py full                 # collect -> process -> embed -> analyze
    python main.py demo                 # synthetic data, no API keys

Environment variables (only needed for the matching stage):
    RAPIDAPI_KEY     — JSearch collector
    APIFY_API_TOKEN  — LinkedIn collector (optional, graceful skip)
    OPENAI_API_KEY   — embed + query
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from src.utils.io import PROJECT_ROOT, bootstrap_data_dirs, load_json, save_json
from src.utils.logger import get_logger

logger = get_logger("main")


def _load_full_config() -> dict[str, Any]:
    """
    Load settings.yaml and merge the JSearch query list from
    search_queries.yaml under the ``search_queries`` key so the
    JSearch collector can read it as a flat list.
    """
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(settings_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    queries_path = PROJECT_ROOT / "config" / "search_queries.yaml"
    if queries_path.exists():
        with open(queries_path, "r", encoding="utf-8") as fh:
            q_doc = yaml.safe_load(fh) or {}
        config["search_queries"] = q_doc.get("queries", [])
    else:
        config["search_queries"] = []
    return config


# =============================================================================
# Commands
# =============================================================================

def cmd_collect(args: argparse.Namespace) -> None:
    """Stage 2 — run every enabled collector listed in settings.yaml."""
    config = _load_full_config()
    sources = config.get("sources", {})
    collectors = []

    if sources.get("jsearch", {}).get("enabled", False):
        from src.collectors import JSearchCollector
        api_key = os.environ.get("RAPIDAPI_KEY", "")
        if api_key:
            collectors.append(JSearchCollector(config=config, api_key=api_key))
        else:
            logger.warning("RAPIDAPI_KEY not set — skipping JSearch")

    if sources.get("hackernews", {}).get("enabled", False):
        from src.collectors import HackerNewsCollector
        collectors.append(HackerNewsCollector(config=config))

    if sources.get("arbeitnow", {}).get("enabled", False):
        from src.collectors import ArbeitnowCollector
        collectors.append(ArbeitnowCollector(config=config))

    if sources.get("berlin_startup_jobs", {}).get("enabled", False):
        from src.collectors import BerlinStartupJobsCollector
        collectors.append(BerlinStartupJobsCollector(config=config))

    if sources.get("eu_ai_act", {}).get("enabled", False):
        from src.collectors import EUAIActCollector
        collectors.append(EUAIActCollector(config=config))

    logger.info("Running %d collectors", len(collectors))
    total = 0
    for collector in collectors:
        try:
            jobs = collector.run()
            total += len(jobs)
        except Exception as exc:
            logger.error(
                "Collector %s failed: %s", collector.__class__.__name__, exc,
            )
    logger.info("Collection complete: %d total raw jobs", total)


def cmd_process(args: argparse.Namespace) -> None:
    """Stage 3 — enrich every raw posting into an EnrichedJob."""
    from src.models import RawJob
    from src.processors import EnrichmentPipeline

    raw_dir = PROJECT_ROOT / "data" / "raw"
    if not raw_dir.exists():
        logger.error("No raw data at %s. Run 'collect' or 'demo' first.", raw_dir)
        return

    all_jobs: list[RawJob] = []
    for json_file in sorted(raw_dir.glob("*.json")):
        data = load_json(json_file)
        if not isinstance(data, list):
            logger.warning("Skipping non-list file: %s", json_file.name)
            continue
        for item in data:
            try:
                all_jobs.append(RawJob(**item))
            except Exception as exc:
                logger.debug(
                    "Skipping invalid record from %s: %s", json_file.name, exc,
                )

    logger.info("Loaded %d raw jobs from %s", len(all_jobs), raw_dir)
    if not all_jobs:
        logger.warning("No raw jobs to process")
        return

    pipeline = EnrichmentPipeline()
    pipeline.enrich(all_jobs)


def cmd_embed(args: argparse.Namespace) -> None:
    """Stage 4 — chunk every enriched job and EU AI Act article, then embed."""
    from src.collectors import load_eu_ai_act_articles
    from src.embeddings import Chunker, Embedder
    from src.models import EnrichedJob
    from src.storage import NumpyVectorStore

    enriched_path = PROJECT_ROOT / "data" / "processed" / "enriched_jobs.json"
    if not enriched_path.exists():
        logger.error("No enriched jobs found. Run 'process' first.")
        return

    data = load_json(enriched_path)
    jobs = [EnrichedJob(**item) for item in data]
    logger.info("Loaded %d enriched jobs", len(jobs))

    chunker = Chunker()
    job_chunks = chunker.chunk_jobs(jobs)

    try:
        articles = load_eu_ai_act_articles()
    except Exception as exc:
        logger.warning("Could not load EU AI Act articles: %s", exc)
        articles = []
    reg_chunks = chunker.chunk_eu_articles(articles) if articles else []

    embedder = Embedder()
    job_chunks = embedder.embed_chunks(job_chunks)
    if reg_chunks:
        reg_chunks = embedder.embed_chunks(reg_chunks)

    store = NumpyVectorStore()
    store.save_job_chunks(job_chunks)
    if reg_chunks:
        store.save_regulation_chunks(reg_chunks)

    ledger_path = PROJECT_ROOT / "data" / "reports" / "cost_ledger.json"
    save_json(embedder.cost_ledger.model_dump(), ledger_path)

    logger.info(
        "Embedding complete: %d job chunks + %d regulation chunks. Cost: %.4f EUR",
        len(job_chunks), len(reg_chunks), embedder.cost_ledger.total_cost_eur,
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    """Stage 6 — produce ``data/reports/analytics.json``."""
    from src.analytics import AnalyticsEngine
    from src.models import EnrichedJob

    enriched_path = PROJECT_ROOT / "data" / "processed" / "enriched_jobs.json"
    if not enriched_path.exists():
        logger.error("No enriched jobs found. Run 'process' first.")
        return

    data = load_json(enriched_path)
    jobs = [EnrichedJob(**item) for item in data]

    cost_summary: dict[str, Any] = {}
    ledger_path = PROJECT_ROOT / "data" / "reports" / "cost_ledger.json"
    if ledger_path.exists():
        ledger = load_json(ledger_path)
        cost_summary = {
            "total_eur": sum(e.get("cost_eur", 0) for e in ledger.get("entries", [])),
            "total_usd": sum(e.get("cost_usd", 0) for e in ledger.get("entries", [])),
            "total_tokens": sum(e.get("tokens_used", 0) for e in ledger.get("entries", [])),
            "budget_eur": ledger.get("budget_eur", 30.0),
        }

    engine = AnalyticsEngine()
    engine.run(jobs, cost_summary=cost_summary)


def cmd_query(args: argparse.Namespace) -> None:
    """Stage 5 — ask the RAG engine one question and print the result."""
    from src.embeddings import Embedder
    from src.rag import RAGEngine
    from src.storage import NumpyVectorStore

    question = args.question
    if not question:
        logger.error("Provide a question: python main.py query 'your question'")
        return

    store = NumpyVectorStore()
    store.load()
    if store.total_chunks == 0:
        logger.error("Vector store is empty. Run 'embed' first.")
        return

    embedder = Embedder()
    engine = RAGEngine(store, embedder)
    result = engine.query(question)

    print()
    print("=" * 60)
    print(f"Question:    {result.query}")
    print(f"Confidence:  {result.confidence}  "
          f"(overall={result.confidence_scores.get('overall', 0):.2f})")
    print(f"Chunks used: {result.num_chunks_used}")
    print("-" * 60)
    print(result.answer)
    print("-" * 60)
    if result.sources_jobs:
        print(f"Job sources:    {', '.join(result.sources_jobs)}")
    if result.sources_legal:
        print(f"Legal sources:  {', '.join(result.sources_legal)}")
    print("=" * 60)


def cmd_status(args: argparse.Namespace) -> None:
    """Print which pipeline stages have run and what's on disk."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_path = PROJECT_ROOT / "data" / "processed" / "enriched_jobs.json"
    embeddings_dir = PROJECT_ROOT / "data" / "embeddings"
    analytics_path = PROJECT_ROOT / "data" / "reports" / "analytics.json"
    ledger_path = PROJECT_ROOT / "data" / "reports" / "cost_ledger.json"

    print()
    print("Berlin AI Talent Radar — Pipeline Status")
    print("=" * 50)

    raw_counts: dict[str, int] = {}
    if raw_dir.exists():
        for f in raw_dir.glob("*.json"):
            try:
                data = load_json(f)
                raw_counts[f.stem] = len(data) if isinstance(data, list) else 0
            except Exception:
                raw_counts[f.stem] = 0
    total_raw = sum(raw_counts.values())
    print(f"\n[Stage 2] Raw jobs:       {total_raw}")
    for source, count in sorted(raw_counts.items()):
        print(f"  - {source:24s} {count}")

    if processed_path.exists():
        data = load_json(processed_path)
        print(f"\n[Stage 3] Enriched jobs:  {len(data)}")
    else:
        print("\n[Stage 3] Enriched jobs:  (not run)")

    npz_files = list(embeddings_dir.glob("*.npz")) if embeddings_dir.exists() else []
    if npz_files:
        print(f"\n[Stage 4] Vector store:   {len(npz_files)} collection(s)")
        for f in sorted(npz_files):
            print(f"  - {f.name}")
    else:
        print("\n[Stage 4] Vector store:   (not run)")

    if analytics_path.exists():
        print("\n[Stage 6] Analytics:      ready")
    else:
        print("\n[Stage 6] Analytics:      (not run)")

    if ledger_path.exists():
        ledger = load_json(ledger_path)
        total_eur = sum(e.get("cost_eur", 0) for e in ledger.get("entries", []))
        budget = ledger.get("budget_eur", 30.0)
        print(f"\n[Budget]  Spent: {total_eur:.4f} EUR / {budget:.2f} EUR")

    print("\n" + "=" * 50)


def cmd_full(args: argparse.Namespace) -> None:
    """Run collect → process → embed → analyze sequentially."""
    logger.info("Running full pipeline")
    cmd_collect(args)
    cmd_process(args)
    cmd_embed(args)
    cmd_analyze(args)
    logger.info("Full pipeline complete")


def cmd_demo(args: argparse.Namespace) -> None:
    """
    Run the pipeline on synthetic sample data — no API keys required.

    Copies every ``data/demo/*.json`` file into ``data/raw/`` (so the
    sample postings look like real collector output), then runs the
    process + analyze stages. The embed stage is skipped because it
    requires ``OPENAI_API_KEY``.
    """
    demo_path = PROJECT_ROOT / "data" / "demo"
    raw_dir = PROJECT_ROOT / "data" / "raw"

    demo_files = list(demo_path.glob("*.json")) if demo_path.exists() else []
    if not demo_files:
        logger.error(
            "No demo data found at %s. Add at least one JSON file there.",
            demo_path,
        )
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    for f in demo_files:
        shutil.copy2(f, raw_dir / f.name)
        logger.info("Copied demo data: %s", f.name)

    cmd_process(args)
    cmd_analyze(args)
    logger.info(
        "Demo complete. Launch the dashboard with: "
        "streamlit run dashboard/app.py"
    )
    logger.info(
        "To enable the RAG chat tab, export OPENAI_API_KEY and run "
        "'python main.py embed'."
    )


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    """Argument parsing + dispatch."""
    parser = argparse.ArgumentParser(
        prog="Berlin AI Talent Radar",
        description="RAG intelligence platform for Berlin's AI job market",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage")

    subparsers.add_parser("collect", help="Stage 2: run all enabled collectors")
    subparsers.add_parser("process", help="Stage 3: enrich raw jobs")
    subparsers.add_parser("embed", help="Stage 4: chunk, embed, persist vectors")
    subparsers.add_parser("analyze", help="Stage 6: produce analytics.json")
    subparsers.add_parser("full", help="Run the full pipeline")
    subparsers.add_parser("demo", help="Run with synthetic sample data")
    subparsers.add_parser("status", help="Show pipeline status")

    query_parser = subparsers.add_parser("query", help="Stage 5: ask the RAG engine")
    query_parser.add_argument("question", type=str, help="Your question")

    args = parser.parse_args()

    bootstrap_data_dirs()

    commands = {
        "collect": cmd_collect,
        "process": cmd_process,
        "embed": cmd_embed,
        "analyze": cmd_analyze,
        "query": cmd_query,
        "status": cmd_status,
        "full": cmd_full,
        "demo": cmd_demo,
    }
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
