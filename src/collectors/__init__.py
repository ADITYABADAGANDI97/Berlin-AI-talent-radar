"""
Collectors package for Berlin AI Talent Radar.

Each collector implements ``BaseCollector`` and is independently
runnable.  The pipeline orchestrator in ``main.py`` calls each
collector's ``.run()`` method.

Available collectors:
- ``JSearchCollector`` — LinkedIn/Indeed/Glassdoor via JSearch RapidAPI
- ``HackerNewsCollector`` — HN "Who is Hiring?" (6-month historical)
- ``ArbeitnowCollector`` — Arbeitnow public JSON API (no auth)
- ``BerlinStartupJobsCollector`` — BSJ HTML scraper
- ``LinkedInApifyCollector`` — LinkedIn via Apify (optional)
- ``EUAIActCollector`` — EU AI Act articles from YAML config
"""

from src.collectors.arbeitnow import ArbeitnowCollector
from src.collectors.base import BaseCollector, CollectorError
from src.collectors.berlin_startup_jobs import BerlinStartupJobsCollector
from src.collectors.eu_ai_act import EUAIActCollector, load_eu_ai_act_articles
from src.collectors.hackernews import HackerNewsCollector
from src.collectors.jsearch import JSearchCollector
from src.collectors.linkedin_apify import LinkedInApifyCollector

__all__ = [
    "BaseCollector",
    "CollectorError",
    "JSearchCollector",
    "HackerNewsCollector",
    "ArbeitnowCollector",
    "BerlinStartupJobsCollector",
    "LinkedInApifyCollector",
    "EUAIActCollector",
    "load_eu_ai_act_articles",
]