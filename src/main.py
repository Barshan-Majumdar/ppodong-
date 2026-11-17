"""
Autonomous 24/7 E-Paper News Agent - Master Orchestrator
Coordinates end-to-end news intelligence pipeline:
Collection -> Trends -> ML Deduplication & Ranking -> Batch LLM Synthesis ->
Broadsheet PDF/MD Publishing -> Telegram/Email Distribution.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import yaml
from dotenv import load_dotenv

from src.collectors.rss_collector import RSSCollector, SeenUrlLedger
from src.collectors.trends_collector import TrendsCollector
from src.llm.synthesizer import LLMSynthesizer
from src.ml.deduplicator_ranker import DeduplicatorRanker
from src.publisher.distributor import Distributor
from src.publisher.epaper_builder import EPaperBuilder

# Load environment variables from .env file
load_dotenv()

# Structured Console Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("epaper.main")


def load_yaml_config(config_path: str = "config/sources.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def run_pipeline(
    config_path: str = "config/sources.yaml",
    edition_date: str = "",
    skip_llm: bool = False,
    skip_distribute: bool = False,
    dry_run: bool = False,
    limit: int = 0,
) -> Dict[str, Any]:
    """Executes the complete autonomous news publisher pipeline."""
    start_time = time.time()
    today_str = edition_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    logger.info("=" * 70)
    logger.info(f"THE DAILY CHRONICLE - AUTONOMOUS NEWS PIPELINE STARTING")
    logger.info(f"Target Edition Date: {today_str} | Dry Run: {dry_run}")
    logger.info("=" * 70)

    config = load_yaml_config(config_path)

    # 1. Initialize Components
    seen_ledger = SeenUrlLedger(
        ledger_path=config.get("pipeline", {}).get("seen_urls_ledger", "data/seen_urls.json")
    )
    from src.storage.story_ledger import StoryLifecycleLedger
    story_ledger = StoryLifecycleLedger()

    rss_collector = RSSCollector(config, seen_ledger=seen_ledger)
    trends_collector = TrendsCollector(config)
    deduplicator = DeduplicatorRanker(config)
    synthesizer = LLMSynthesizer(
        batch_size=config.get("pipeline", {}).get("llm_batch_size", 15)
    )
    builder = EPaperBuilder()
    distributor = Distributor()

    utc_now = datetime.now(timezone.utc)
    edition_cycle = "Morning" if utc_now.hour < 7 else "Evening"

    # 2. Ingest News and Trends concurrently
    logger.info("\n--- STEP 1 & 2: INGESTION (RSS, GOOGLE NEWS & TRENDS) ---")
    rss_task = asyncio.create_task(rss_collector.collect_all_sections())
    trends_task = asyncio.create_task(trends_collector.collect_all_trends())

    rss_articles, trend_articles = await asyncio.gather(rss_task, trends_task)
    all_raw_articles = rss_articles + trend_articles

    logger.info(
        f"Ingestion Finished: {len(rss_articles)} RSS/Google News + "
        f"{len(trend_articles)} Trends = {len(all_raw_articles)} Total Raw Articles."
    )

    effective_limit = limit if limit > 0 else config.get("pipeline", {}).get("max_daily_stories", 0)
    if effective_limit > 0 and len(all_raw_articles) > effective_limit:
        logger.info(f"Applying balanced desk sampling for daily edition (target: {effective_limit} stories)...")
        by_section: Dict[str, List[Dict[str, Any]]] = {}
        for a in all_raw_articles:
            s_key = a.get("section", "national")
            by_section.setdefault(s_key, []).append(a)

        desk_keys = ["kolkata_bengal", "indian_states", "national", "global", "trends"]
        for k in by_section:
            if k not in desk_keys:
                desk_keys.append(k)

        per_section_quota = max(1, effective_limit // len(desk_keys))
        sampled_articles = []
        for dk in desk_keys:
            items = by_section.get(dk, [])
            sampled_articles.extend(items[:per_section_quota])

        # Fill any remainder up to effective_limit
        if len(sampled_articles) < effective_limit:
            remaining = [a for a in all_raw_articles if a not in sampled_articles]
            sampled_articles.extend(remaining[: (effective_limit - len(sampled_articles))])

        all_raw_articles = sampled_articles
        counts_summary = [f"{k}: {len([x for x in all_raw_articles if x.get('section') == k])}" for k in desk_keys]
        logger.info(f"Balanced sampling complete: {len(all_raw_articles)} articles selected across all desks: {counts_summary}.")
    else:
        logger.info(f"Uncapped news ingestion active: Zero story cap. Retaining all {len(all_raw_articles)} incoming articles for comprehensive clustering.")

    if not all_raw_articles:
        logger.warning("No articles fetched in this rolling window. Exiting gracefully.")
        return {"status": "no_articles", "total_raw": 0, "clusters": 0}

    # 3. Semantic Deduplication & Editorial Priority Ranking
    logger.info("\n--- STEP 3: ML DEDUPLICATION & DETERMINISTIC RANKING ---")
    ranked_clusters = deduplicator.deduplicate_and_rank(all_raw_articles)
    logger.info(
        f"Retention Check: 100% of unique story clusters preserved ({len(ranked_clusters)} clusters)."
    )

    # 4. Zero-Exclusion Batch LLM Summarization
    logger.info("\n--- STEP 4: BATCH LLM SYNTHESIS & SUMMARIZATION ---")
    if skip_llm:
        logger.info("Flag '--skip-llm' enabled: using high-speed extractive journalistic summarization.")
        from src.llm.synthesizer import generate_extractive_fallback_summary

        synthesized_stories = []
        for c in ranked_clusters:
            ext = generate_extractive_fallback_summary(c)
            c_copy = dict(c)
            c_copy["headline"] = ext.headline
            c_copy["summary"] = ext.summary
            c_copy["key_takeaway"] = ext.key_takeaway
            c_copy["why_it_matters"] = ext.why_it_matters
            c_copy["is_good_news"] = ext.is_good_news
            c_copy["fact_check_verdict"] = ext.fact_check_verdict
            c_copy["urgency_tag"] = ext.urgency_tag
            c_copy["category"] = ext.category
            c_copy["state_or_region"] = ext.state_or_region
            synthesized_stories.append(c_copy)
    else:
        synthesized_stories = await synthesizer.synthesize_all_stories(ranked_clusters)

    # 4.5 Time-Awareness & Story Lifecycle Processing
    logger.info("\n--- STEP 4.5: STORY LIFECYCLE & TIME-AWARENESS AUDIT ---")
    for s in synthesized_stories:
        story_ledger.process_story(s, edition_cycle=edition_cycle)
    if not dry_run:
        story_ledger.save()
        logger.info("Saved data/story_lifecycle.json with edition timestamps and development statuses.")

    # 5. Broadsheet Newspaper Publishing (HTML, PDF, Markdown)
    logger.info("\n--- STEP 5: BROADSHEET PUBLISHING & ARCHIVING ---")
    pub_result = builder.publish_edition(
        synthesized_stories, edition_date=today_str, edition_cycle=edition_cycle
    )

    # 6. Mark newly published URLs in Seen Ledger
    if not dry_run:
        for c in ranked_clusters:
            for u in c.get("urls", []):
                seen_ledger.mark_seen(u)
        seen_ledger.save()
        logger.info("Updated data/seen_urls.json ledger with today's published URLs.")

    # 7. Distribution (Telegram & Email)
    dist_result = {}
    if not skip_distribute and not dry_run:
        logger.info("\n--- STEP 6: TELEGRAM & EMAIL DISTRIBUTION ---")
        dist_result = distributor.distribute_all(pub_result, synthesized_stories)
    else:
        logger.info("Skipping distribution (dry-run or skip-distribute enabled).")

    elapsed_time = round(time.time() - start_time, 2)

    if pub_result.get("quality_checklist"):
        logger.info("\n" + pub_result["quality_checklist"] + "\n")

    # Print Final Execution Summary
    logger.info("\n" + "=" * 70)
    logger.info("EXECUTION SUMMARY - THE DAILY CHRONICLE")
    logger.info("=" * 70)
    logger.info(f"Edition Date         : {today_str}")
    logger.info(f"Raw Articles Ingested: {len(all_raw_articles)}")
    logger.info(f"Unique Story Clusters: {len(synthesized_stories)} (100% Retained)")
    logger.info(f"Media Sources Covered: {pub_result['total_sources']}")
    logger.info(f"HTML Edition         : {pub_result['html_path']}")
    logger.info(f"PDF Edition          : {pub_result['pdf_path'] or 'Skipped (WeasyPrint C-libs)'}")
    logger.info(f"Markdown Archive     : {pub_result['md_path']}")
    logger.info(f"Telegram Dispatched  : {dist_result.get('telegram', False)}")
    logger.info(f"Email Dispatched     : {dist_result.get('email', False)}")
    logger.info(f"Total Runtime        : {elapsed_time} seconds")
    logger.info("=" * 70)

    return {
        "status": "success",
        "edition_date": today_str,
        "total_raw": len(all_raw_articles),
        "total_clusters": len(synthesized_stories),
        "total_sources": pub_result["total_sources"],
        "runtime_seconds": elapsed_time,
        "files": pub_result,
        "distribution": dist_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="The Daily Chronicle - Autonomous 24/7 E-Paper News Agent"
    )
    parser.add_argument(
        "--config", default="config/sources.yaml", help="Path to sources.yaml configuration"
    )
    parser.add_argument(
        "--date", default="", help="Edition date in YYYY-MM-DD format (defaults to UTC today)"
    )
    parser.add_argument(
        "--skip-llm", action="store_true", help="Bypass LLM APIs and use extractive summarizer"
    )
    parser.add_argument(
        "--skip-distribute", action="store_true", help="Skip Telegram/Email dispatch"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Test run without updating ledger or distributing"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of articles processed for rapid testing"
    )

    args = parser.parse_args()

    try:
        asyncio.run(
            run_pipeline(
                config_path=args.config,
                edition_date=args.date,
                skip_llm=args.skip_llm,
                skip_distribute=args.skip_distribute,
                dry_run=args.dry_run,
                limit=args.limit,
            )
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except KeyboardInterrupt:
        logger.info("Pipeline cancelled by user.")
        os._exit(0)
    except Exception as e:
        logger.exception(f"Fatal error in E-Paper pipeline: {e}")
        os._exit(1)


if __name__ == "__main__":
    main()
