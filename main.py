"""
Berlin AI Talent Radar — CLI Orchestrator
==========================================
Entry point for the full pipeline. Run stages individually or
the complete pipeline end-to-end.

Usage::

    python main.py collect       # Stage 2: Collect raw jobs from all sources
    python main.py process       # Stage 3: Enrich raw jobs (skills, scores, governance)
    python main.py embed         # Stage 4: Chunk + embed + store vectors
    python main.py analyze       # Stage 5: Run analytics engine
    python main.py query "..."   # Stage 4: Ask the RAG engine a question
    python main.py status        # Show pipeline status
    python main.py full          # Run full pipeline: collect → process → embed → analyze
    python main.py demo          # Run with sample/demo data (no API keys needed)
"""

import argparse
import json
import sys
from pathlib import Path

from src.utils.Io import (
    PROJECT_ROOT, bootstrap_data_dirs, load_json, save_json,
)
from src.utils.logger import get_logger

logger = get_logger("main")


def cmd_collect(args: argparse.Namespace) -> None:
    """Stage 2: Run all enabled collectors."""
    import yaml

    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    from src.collectors.jsearch import JSearchCollector
    from src.collectors.hackernews import HackerNewsCollector
    from src.collectors.arbeitnow import ArbeitnowCollector
    from src.collectors.berlin_startup_jobs import BerlinStartupJobsCollector
    from src.collectors.eu_ai_act import EUAIActCollector

    collectors = []
    sources = config.get("sources", {})

    if sources.get("jsearch", {}).get("enabled", False):
        import os
        api_key = os.environ.get("RAPIDAPI_KEY", "")
        if api_key:
            collectors.append(JSearchCollector(api_key=api_key, config=config))
        else:
            logger.warning("RAPIDAPI_KEY not set — skipping JSearch")

    if sources.get("hackernews", {}).get("enabled", False):
        collectors.append(HackerNewsCollector(config=config))

    if sources.get("arbeitnow", {}).get("enabled", False):
        collectors.append(ArbeitnowCollector(config=config))

    if sources.get("berlin_startup_jobs", {}).get("enabled", False):
        collectors.append(BerlinStartupJobsCollector(config=config))

    if sources.get("eu_ai_act", {}).get("enabled", False):
        collectors.append(EUAIActCollector(config=config))

    logger.info("Running %d collectors", len(collectors))
    total = 0
    for collector in collectors:
        jobs = collector.run()
        total += len(jobs)

    logger.info("Collection complete: %d total raw jobs", total)


def cmd_process(args: argparse.Namespace) -> None:
    """Stage 3: Process raw jobs into enriched jobs."""
    from src.models import RawJob
    from src.processors import EnrichmentPipeline

    # Load all raw JSON files
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if not raw_dir.exists():
        logger.error("No raw data found at %s. Run 'collect' first.", raw_dir)
        return

    all_jobs: list[RawJob] = []
    for json_file in sorted(raw_dir.glob("*.json")):
        data = load_json(json_file)
        for item in data:
            try:
                all_jobs.append(RawJob(**item))
            except Exception as exc:
                logger.debug("Skipping invalid job from %s: %s", json_file.name, exc)

    logger.info("Loaded %d raw jobs from %s", len(all_jobs), raw_dir)

    if not all_jobs:
        logger.warning("No raw jobs to process")
        return

    pipeline = EnrichmentPipeline()
    pipeline.enrich(all_jobs)


def cmd_embed(args: argparse.Namespace) -> None:
    """Stage 4: Chunk, embed, and store vectors."""
    from src.models import EnrichedJob
    from src.embeddings import Chunker, Embedder
    from src.storage import NumpyVectorStore
    from src.collectors.eu_ai_act import load_eu_ai_act_articles

    # Load enriched jobs
    enriched_path = PROJECT_ROOT / "data" / "processed" / "enriched_jobs.json"
    if not enriched_path.exists():
        logger.error("No enriched jobs found. Run 'process' first.")
        return

    data = load_json(enriched_path)
    jobs = [EnrichedJob(**item) for item in data]
    logger.info("Loaded %d enriched jobs", len(jobs))

    # Chunk
    chunker = Chunker()
    job_chunks = chunker.chunk_jobs(jobs)

    # EU AI Act articles
    articles = load_eu_ai_act_articles()
    reg_chunks = chunker.chunk_eu_articles(articles)

    # Embed
    embedder = Embedder()
    job_chunks = embedder.embed_chunks(job_chunks)
    reg_chunks = embedder.embed_chunks(reg_chunks)

    # Store
    store = NumpyVectorStore()
    store.save_job_chunks(job_chunks)
    store.save_regulation_chunks(reg_chunks)

    # Save cost ledger
    ledger_path = PROJECT_ROOT / "data" / "reports" / "cost_ledger.json"
    save_json(embedder.cost_ledger.model_dump(), ledger_path)

    logger.info(
        "Embedding complete: %d job chunks + %d regulation chunks. Cost: %.4f EUR",
        len(job_chunks), len(reg_chunks), embedder.cost_ledger.total_cost_eur,
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    """Stage 5: Run analytics engine."""
    from src.models import EnrichedJob
    from src.analytics import AnalyticsEngine

    enriched_path = PROJECT_ROOT / "data" / "processed" / "enriched_jobs.json"
    if not enriched_path.exists():
        logger.error("No enriched jobs found. Run 'process' first.")
        return

    data = load_json(enriched_path)
    jobs = [EnrichedJob(**item) for item in data]

    engine = AnalyticsEngine()
    engine.run(jobs)


def cmd_query(args: argparse.Namespace) -> None:
    """Ask the RAG engine a question."""
    from src.embeddings import Embedder
    from src.storage import NumpyVectorStore
    from src.rag import RAGEngine

    question = args.question
    if not question:
        logger.error("Please provide a question: python main.py query 'your question'")
        return

    # Load vector store
    store = NumpyVectorStore()
    store.load()

    if store.total_chunks == 0:
        logger.error("Vector store is empty. Run 'embed' first.")
        return

    # Initialize RAG
    embedder = Embedder()
    engine = RAGEngine(store, embedder)

    # Query
    result = engine.query(question)

    # Display result
    print("\n" + "=" * 60)
    print(f"Question: {result.query}")
    print(f"Confidence: {result.confidence}")
    print(f"Chunks used: {result.num_chunks_used}")
    print("-" * 60)
    print(result.answer)
    print("-" * 60)
    if result.sources_jobs:
        print(f"Job sources: {', '.join(result.sources_jobs)}")
    if result.sources_legal:
        print(f"Legal sources: {', '.join(result.sources_legal)}")
    print("=" * 60)


def cmd_status(args: argparse.Namespace) -> None:
    """Show pipeline status."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_path = PROJECT_ROOT / "data" / "processed" / "enriched_jobs.json"
    embeddings_dir = PROJECT_ROOT / "data" / "embeddings"

    print("\n Berlin AI Talent Radar — Pipeline Status")
    print("=" * 50)

    # Raw jobs
    raw_counts: dict[str, int] = {}
    if raw_dir.exists():
        for f in raw_dir.glob("*.json"):
            data = load_json(f)
            raw_counts[f.stem] = len(data) if isinstance(data, list) else 0
    total_raw = sum(raw_counts.values())
    print(f"\n[Stage 2] Raw jobs:       {total_raw}")
    for source, count in raw_counts.items():
        print(f"  - {source}: {count}")

    # Enriched jobs
    if processed_path.exists():
        data = load_json(processed_path)
        print(f"\n[Stage 3] Enriched jobs:  {len(data)}")
    else:
        print("\n[Stage 3] Enriched jobs:  (not run)")

    # Embeddings
    npz_files = list(embeddings_dir.glob("*.npz")) if embeddings_dir.exists() else []
    if npz_files:
        print(f"\n[Stage 4] Vector store:   {len(npz_files)} collection(s)")
        for f in npz_files:
            print(f"  - {f.name}")
    else:
        print("\n[Stage 4] Vector store:   (not run)")

    # Analytics
    analytics_path = PROJECT_ROOT / "data" / "reports" / "analytics.json"
    if analytics_path.exists():
        print("\n[Stage 5] Analytics:      ready")
    else:
        print("\n[Stage 5] Analytics:      (not run)")

    # Cost
    ledger_path = PROJECT_ROOT / "data" / "reports" / "cost_ledger.json"
    if ledger_path.exists():
        ledger = load_json(ledger_path)
        total_eur = sum(e.get("cost_eur", 0) for e in ledger.get("entries", []))
        budget = ledger.get("budget_eur", 30.0)
        print(f"\n[Budget] Spent: {total_eur:.4f} EUR / {budget:.2f} EUR")

    print("\n" + "=" * 50)


def cmd_full(args: argparse.Namespace) -> None:
    """Run full pipeline: collect → process → embed → analyze."""
    logger.info("Running full pipeline")
    cmd_collect(args)
    cmd_process(args)
    cmd_embed(args)
    cmd_analyze(args)
    logger.info("Full pipeline complete")


def cmd_demo(args: argparse.Namespace) -> None:
    """Run with demo/sample data (no API keys needed for processing)."""
    demo_path = PROJECT_ROOT / "data" / "demo"
    raw_dir = PROJECT_ROOT / "data" / "raw"

    if not demo_path.exists() or not list(demo_path.glob("*.json")):
        logger.error(
            "No demo data found at %s. "
            "Place sample JSON files in data/demo/ first.",
            demo_path,
        )
        return

    # Copy demo data to raw
    import shutil
    raw_dir.mkdir(parents=True, exist_ok=True)
    for f in demo_path.glob("*.json"):
        shutil.copy2(f, raw_dir / f.name)
        logger.info("Copied demo data: %s", f.name)

    # Run processing and analytics (no embedding — requires API key)
    cmd_process(args)
    cmd_analyze(args)
    logger.info("Demo complete. Run 'embed' separately if you have an OPENAI_API_KEY.")


def main() -> None:
    """Parse arguments and dispatch to command handler."""
    parser = argparse.ArgumentParser(
        prog="Berlin AI Talent Radar",
        description="RAG intelligence platform for Berlin's AI job market",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline command")

    subparsers.add_parser("collect", help="Run all collectors")
    subparsers.add_parser("process", help="Enrich raw jobs")
    subparsers.add_parser("embed", help="Chunk, embed, and store vectors")
    subparsers.add_parser("analyze", help="Run analytics engine")
    subparsers.add_parser("full", help="Run full pipeline")
    subparsers.add_parser("demo", help="Run with demo data")
    subparsers.add_parser("status", help="Show pipeline status")

    query_parser = subparsers.add_parser("query", help="Ask the RAG engine")
    query_parser.add_argument("question", type=str, help="Your question")

    args = parser.parse_args()

    # Bootstrap directories
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


if __name__ == "__main__":
    main()
