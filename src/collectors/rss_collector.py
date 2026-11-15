"""
Autonomous 24/7 E-Paper News Agent - RSS & Google News Collector
Handles concurrent RSS/Google News fetching, redirect resolution, 
timestamp filtering (rolling window), and clean article text extraction via trafilatura.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

import aiohttp
import feedparser
from bs4 import BeautifulSoup
import trafilatura

logger = logging.getLogger("epaper.collector.rss")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def extract_domain(url: str) -> str:
    """Extract clean domain name from URL."""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def decode_google_news_url_fallback(google_url: str) -> Optional[str]:
    """
    Attempt deterministic base64 decoding of protobuf-based Google News article URLs.
    Google News URLs format: https://news.google.com/rss/articles/CBMi...
    """
    try:
        match = re.search(r"articles/([a-zA-Z0-9_\-]+)", google_url)
        if not match:
            return None
        token = match.group(1)
        # Pad base64
        padded = token + "=" * ((4 - len(token) % 4) % 4)
        raw_bytes = base64.urlsafe_b64decode(padded.encode("ascii", errors="ignore"))
        # Search for embedded http or https URLs inside decoded protobuf bytes
        urls = re.findall(rb"https?://[a-zA-Z0-9.\-_~:/?#\[\]@!$&'()*+,;=%]+", raw_bytes)
        for u in urls:
            decoded = u.decode("utf-8", errors="ignore")
            if "news.google.com" not in decoded and len(decoded) > 12:
                # Clean trailing binary garbage if present
                clean_url = re.split(r"[\x00-\x1f\x7f-\xff]", decoded)[0]
                return clean_url
    except Exception:
        pass
    return None


class SeenUrlLedger:
    """Maintains a rolling 7-day ledger of previously processed article URLs."""

    def __init__(self, ledger_path: str = "data/seen_urls.json", retention_days: int = 7):
        self.ledger_path = ledger_path
        self.retention_days = retention_days
        self.seen: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.seen = data.get("urls", {})
            except Exception as e:
                logger.warning(f"Failed to load seen URLs ledger: {e}. Starting fresh.")
                self.seen = {}
        else:
            self.seen = {}

    def is_seen(self, url: str) -> bool:
        return url in self.seen

    def mark_seen(self, url: str, timestamp_iso: Optional[str] = None) -> None:
        if not timestamp_iso:
            timestamp_iso = datetime.now(timezone.utc).isoformat()
        self.seen[url] = timestamp_iso

    def save(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        cleaned: Dict[str, str] = {}
        for u, ts in self.seen.items():
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt >= cutoff:
                    cleaned[u] = ts
            except Exception:
                cleaned[u] = ts

        self.seen = cleaned
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        try:
            with open(self.ledger_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "urls": self.seen,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Error saving seen URLs ledger: {e}")


class RSSCollector:
    """Asynchronous collector for RSS and Google News feeds."""

    def __init__(
        self,
        config: Dict[str, Any],
        seen_ledger: Optional[SeenUrlLedger] = None,
        concurrency_limit: int = 20,
    ):
        self.config = config
        self.pipeline_cfg = config.get("pipeline", {})
        self.window_hours = self.pipeline_cfg.get("rolling_window_hours", 26)
        self.max_text_chars = self.pipeline_cfg.get("max_text_extract_chars", 1800)
        self.authority_domains = set(
            config.get("authority_domains", {}).get("high", [])
        )
        ledger_path = self.pipeline_cfg.get("seen_urls_ledger", "data/seen_urls.json")
        self.seen_ledger = seen_ledger or SeenUrlLedger(ledger_path=ledger_path)
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.headers = {"User-Agent": DEFAULT_USER_AGENT}

    async def fetch_feed_xml(
        self, session: aiohttp.ClientSession, feed_name: str, url: str
    ) -> Optional[str]:
        """Fetch raw XML/RSS content with timeout and error handling."""
        async with self.semaphore:
            try:
                timeout = aiohttp.ClientTimeout(total=18)
                async with session.get(url, headers=self.headers, timeout=timeout) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    else:
                        logger.warning(f"Feed '{feed_name}' returned HTTP {resp.status}")
                        return None
            except Exception as e:
                logger.warning(f"Error fetching feed '{feed_name}' from {url}: {e}")
                return None

    async def resolve_canonical_url(
        self, session: aiohttp.ClientSession, original_url: str
    ) -> str:
        """Resolve Google News redirect or tracking links to real publisher destination."""
        if "news.google.com" not in original_url:
            return original_url

        # 1. Try deterministic decoding first
        decoded = decode_google_news_url_fallback(original_url)
        if decoded and decoded.startswith("http"):
            return decoded

        # 2. Try network HEAD/GET redirect resolution
        async with self.semaphore:
            try:
                timeout = aiohttp.ClientTimeout(total=8)
                async with session.get(
                    original_url,
                    headers=self.headers,
                    allow_redirects=True,
                    timeout=timeout,
                ) as resp:
                    final_url = str(resp.url)
                    if "news.google.com" not in final_url and final_url.startswith("http"):
                        return final_url
                    # Sometimes the destination is inside a meta refresh or canonical tag
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")
                    canonical = soup.find("link", rel="canonical")
                    if canonical and canonical.get("href"):
                        href = canonical["href"]
                        if href.startswith("http") and "news.google.com" not in href:
                            return href
            except Exception:
                pass

        return original_url

    async def extract_article_body(
        self,
        session: aiohttp.ClientSession,
        url: str,
        fallback_summary: str,
        title: str,
    ) -> str:
        """Extract full article text using trafilatura with BeautifulSoup & summary fallback."""
        cleaned_fallback = BeautifulSoup(fallback_summary or "", "html.parser").get_text(
            separator=" ", strip=True
        )

        # Skip scraping on pure aggregator/redirect domains if not resolved
        if "news.google.com" in url or not url.startswith("http"):
            return cleaned_fallback if len(cleaned_fallback) > 30 else f"{title}. {cleaned_fallback}"

        async with self.semaphore:
            try:
                timeout = aiohttp.ClientTimeout(total=10)
                async with session.get(url, headers=self.headers, timeout=timeout) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        extracted = trafilatura.extract(
                            html,
                            favor_precision=True,
                            include_comments=False,
                            include_tables=False,
                        )
                        if extracted and len(extracted.strip()) >= 120:
                            # Truncate safely
                            return extracted.strip()[: self.max_text_chars]
            except Exception:
                pass

        # Fallback to description / summary
        if len(cleaned_fallback) >= 40:
            return cleaned_fallback[: self.max_text_chars]
        return f"{title}. {cleaned_fallback}".strip()[: self.max_text_chars]

    def parse_entry_time(self, entry: Any, now_utc: datetime) -> datetime:
        """Parse entry publication time or return now_utc if missing."""
        for field in ("published_parsed", "updated_parsed", "created_parsed"):
            val = getattr(entry, field, None) or entry.get(field)
            if val:
                try:
                    return datetime.fromtimestamp(time.mktime(val), tz=timezone.utc)
                except Exception:
                    pass
        return now_utc

    async def process_feed_entries(
        self,
        session: aiohttp.ClientSession,
        feed_config: Dict[str, Any],
        section_key: str,
        raw_xml: str,
        now_utc: datetime,
        cutoff_utc: datetime,
    ) -> List[Dict[str, Any]]:
        """Parse XML feed items, apply time filtering, resolve URLs and extract content."""
        feed = feedparser.parse(raw_xml)
        feed_name = feed_config.get("name", "Unknown Feed")
        default_region = feed_config.get("region", section_key)

        items_to_fetch = []
        for entry in feed.entries:
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            if not title or not link:
                continue

            entry_time = self.parse_entry_time(entry, now_utc)
            if entry_time < cutoff_utc:
                continue

            # Check seen ledger
            if self.seen_ledger.is_seen(link):
                continue

            summary = entry.get("summary") or entry.get("description") or ""
            items_to_fetch.append((entry, title, link, entry_time, summary))

        results: List[Dict[str, Any]] = []

        async def _fetch_single_item(item_tuple: tuple) -> Optional[Dict[str, Any]]:
            entry, title, link, entry_time, summary = item_tuple
            try:
                canonical_url = await self.resolve_canonical_url(session, link)
                domain = extract_domain(canonical_url) or extract_domain(link)
                is_high_auth = domain in self.authority_domains

                body_text = await self.extract_article_body(
                    session, canonical_url, summary, title
                )

                # Derive clean source name
                source_name = getattr(entry, "source", None)
                if isinstance(source_name, dict):
                    source_str = source_name.get("title") or domain
                else:
                    source_str = domain or feed_name

                # Clean Title if Google News appends " - Publisher"
                clean_title = re.sub(r"\s+-\s+[^-]+$", "", title) if "news.google.com" in link else title

                article_id = hashlib.sha256(
                    f"{clean_title}_{canonical_url}".encode("utf-8")
                ).hexdigest()[:16]

                lead_text = body_text[:300].replace("\n", " ").strip()

                return {
                    "id": article_id,
                    "title": clean_title,
                    "url": link,
                    "canonical_url": canonical_url,
                    "source": source_str,
                    "domain": domain,
                    "published_at": entry_time.isoformat(),
                    "timestamp": entry_time.timestamp(),
                    "section": section_key,
                    "region": default_region,
                    "text": body_text,
                    "lead": lead_text,
                    "is_high_authority": is_high_auth,
                }
            except Exception as e:
                logger.debug(f"Error processing item '{title[:30]}': {e}")
                return None

        tasks = [_fetch_single_item(item) for item in items_to_fetch]
        if tasks:
            extracted_items = await asyncio.gather(*tasks, return_exceptions=True)
            for res in extracted_items:
                if isinstance(res, dict):
                    results.append(res)

        return results

    async def collect_all_sections(self) -> List[Dict[str, Any]]:
        """Collect articles from all configured sections concurrently."""
        now_utc = datetime.now(timezone.utc)
        cutoff_utc = now_utc - timedelta(hours=self.window_hours)
        logger.info(
            f"Starting RSS collection. Rolling window: {self.window_hours}h "
            f"(since {cutoff_utc.strftime('%Y-%m-%d %H:%M UTC')})"
        )

        all_articles: List[Dict[str, Any]] = []
        sections_config = self.config.get("sections", {})

        connector = aiohttp.TCPConnector(limit=30, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Step 1: Fetch all feed XMLs
            feed_fetch_tasks = []
            feed_metadata = []

            for section_key, sec_data in sections_config.items():
                feeds = sec_data.get("feeds", [])
                for feed_item in feeds:
                    feed_name = feed_item.get("name")
                    feed_url = feed_item.get("url")
                    feed_fetch_tasks.append(
                        self.fetch_feed_xml(session, feed_name, feed_url)
                    )
                    feed_metadata.append(
                        (section_key, feed_item, sec_data.get("default_region", section_key))
                    )

            logger.info(f"Dispatching concurrent fetch for {len(feed_fetch_tasks)} feeds...")
            feed_xml_results = await asyncio.gather(*feed_fetch_tasks, return_exceptions=True)

            # Step 2: Parse items and extract article contents
            entry_processing_tasks = []
            for i, raw_xml in enumerate(feed_xml_results):
                if isinstance(raw_xml, str) and len(raw_xml) > 50:
                    sec_key, feed_item, def_reg = feed_metadata[i]
                    f_cfg = dict(feed_item)
                    f_cfg["region"] = def_reg
                    entry_processing_tasks.append(
                        self.process_feed_entries(
                            session, f_cfg, sec_key, raw_xml, now_utc, cutoff_utc
                        )
                    )

            if entry_processing_tasks:
                section_results = await asyncio.gather(
                    *entry_processing_tasks, return_exceptions=True
                )
                for res in section_results:
                    if isinstance(res, list):
                        all_articles.extend(res)

        logger.info(f"RSS Collection complete. Ingested {len(all_articles)} raw valid articles.")
        return all_articles
