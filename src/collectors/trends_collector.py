"""
Autonomous 24/7 E-Paper News Agent - Trends & Social Intelligence Collector
Fetches daily trending searches in India (Google Trends RSS), Indian Express viral trends,
and top community discussions from Indian and Kolkata subreddits.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp
import feedparser

logger = logging.getLogger("epaper.collector.trends")

REDDIT_USER_AGENT = (
    "AutoEPaperBot/1.0 (by /u/NewsIntelligenceBot; contact: news-agent@local.net)"
)
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


class TrendsCollector:
    """Collects viral trends and social sentiment from Google Trends and Reddit."""

    def __init__(self, config: Dict[str, Any], concurrency_limit: int = 6):
        self.config = config
        self.trends_cfg = config.get("trends_sources", {})
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    async def fetch_google_trends(
        self, session: aiohttp.ClientSession, url: str
    ) -> List[Dict[str, Any]]:
        """Fetch and parse Google Trends daily RSS feed (geo=IN)."""
        results: List[Dict[str, Any]] = []
        headers = {"User-Agent": BROWSER_USER_AGENT}

        async with self.semaphore:
            try:
                timeout = aiohttp.ClientTimeout(total=15)
                async with session.get(url, headers=headers, timeout=timeout) as resp:
                    if resp.status != 200:
                        logger.warning(f"Google Trends RSS returned HTTP {resp.status}")
                        return results
                    xml_content = await resp.text()

                # Parse XML using ElementTree
                root = ET.fromstring(xml_content)
                # Namespaces typically used by Google Trends RSS
                namespaces = {"ht": "https://trends.google.com/trending/rss"}

                channel = root.find("channel")
                if channel is None:
                    return results

                now_iso = datetime.now(timezone.utc).isoformat()
                now_ts = datetime.now(timezone.utc).timestamp()

                for item in channel.findall("item"):
                    title_elem = item.find("title")
                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                    if not title:
                        continue

                    lower_title = title.lower()
                    # Filter out non-journalistic spam trends (horoscopes, lotteries, etc.)
                    if any(
                        kw in lower_title
                        for kw in [
                            "horoscope",
                            "rashifal",
                            "astrology",
                            "zodiac",
                            "kundali",
                            "lottery",
                            "satta",
                            "lucky number",
                        ]
                    ):
                        continue

                    traffic_elem = item.find("ht:approx_traffic", namespaces)
                    traffic = traffic_elem.text.strip() if traffic_elem is not None and traffic_elem.text else "Spike"

                    # Collect linked news snippets
                    news_snippets = []
                    source_names = ["Google Trends"]
                    headline_candidate = ""
                    news_items = item.findall("ht:news_item", namespaces)
                    for n_item in news_items:
                        n_title_elem = n_item.find("ht:news_item_title", namespaces)
                        n_snippet_elem = n_item.find("ht:news_item_snippet", namespaces)
                        n_source_elem = n_item.find("ht:news_item_source", namespaces)

                        n_title = n_title_elem.text.strip() if n_title_elem is not None and n_title_elem.text else ""
                        n_snippet = (
                            n_snippet_elem.text.strip()
                            if n_snippet_elem is not None and n_snippet_elem.text
                            else ""
                        )
                        n_source = n_source_elem.text.strip() if n_source_elem is not None and n_source_elem.text else ""

                        # Clean up HTML entities
                        n_title = re.sub(r"&[a-zA-Z0-9#]+;", " ", n_title).strip()
                        n_snippet = re.sub(r"&[a-zA-Z0-9#]+;", " ", n_snippet).strip()

                        if n_title and not headline_candidate and len(n_title) > 10:
                            headline_candidate = n_title

                        if n_title:
                            news_snippets.append(f"{n_title}: {n_snippet}")
                        if n_source and n_source not in source_names:
                            source_names.append(n_source)

                    display_title = headline_candidate or f"Viral Topic: {title.title()}"
                    body_text = (
                        f"Search Volume: {traffic}. "
                        + (" | ".join(news_snippets) if news_snippets else f"Trending search in India: {title}.")
                    )
                    body_text = re.sub(r"\s+", " ", body_text).strip()

                    trend_id = hashlib.sha256(f"trend_google_{title}".encode("utf-8")).hexdigest()[:16]

                    results.append(
                        {
                            "id": trend_id,
                            "title": display_title,
                            "url": f"https://www.google.com/search?q={title.replace(' ', '+')}",
                            "canonical_url": f"https://www.google.com/search?q={title.replace(' ', '+')}",
                            "source": ", ".join(source_names[:3]),
                            "domain": "trends.google.com",
                            "published_at": now_iso,
                            "timestamp": now_ts,
                            "section": "trends",
                            "region": "trends",
                            "text": body_text[:1200],
                            "lead": body_text[:300],
                            "is_high_authority": False,
                            "is_trend": True,
                        }
                    )
            except Exception as e:
                logger.warning(f"Error parsing Google Trends RSS: {e}")

        logger.info(f"Google Trends: Extracted {len(results)} viral topics.")
        return results

    async def fetch_reddit_posts(
        self, session: aiohttp.ClientSession, subreddit: str, url: str
    ) -> List[Dict[str, Any]]:
        """Fetch top daily discussions from Indian subreddits via JSON or RSS fallback."""
        results: List[Dict[str, Any]] = []
        headers = {"User-Agent": BROWSER_USER_AGENT}

        async with self.semaphore:
            # 1. Try JSON endpoint first
            try:
                timeout = aiohttp.ClientTimeout(total=12)
                async with session.get(url, headers=headers, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        children = data.get("data", {}).get("children", [])
                        now_iso = datetime.now(timezone.utc).isoformat()
                        now_ts = datetime.now(timezone.utc).timestamp()

                        for child in children:
                            post = child.get("data", {})
                            if post.get("stickied", False):
                                continue

                            title = post.get("title", "").strip()
                            if not title:
                                continue

                            # Filter out non-newsworthy routine social noise: classifieds, buy/sell, room searches, megathreads
                            lower_title = title.lower()
                            if any(
                                kw in lower_title
                                for kw in [
                                    "classifieds",
                                    "buy/sell",
                                    "megathread",
                                    "weekly discussion",
                                    "monthly discussion",
                                    "flatmate",
                                    "looking for flat",
                                    "roommate",
                                    "rent in",
                                    "selling my",
                                    "buying a",
                                ]
                            ):
                                continue

                            score = post.get("score", 0)
                            num_comments = post.get("num_comments", 0)
                            selftext = post.get("selftext", "").strip()
                            permalink = f"https://reddit.com{post.get('permalink', '')}"

                            cleaned_text = re.sub(r"http\S+", "", selftext).strip()
                            body = (
                                f"Engagement: {score} upvotes, {num_comments} comments. "
                                f"{cleaned_text if len(cleaned_text) > 20 else title}"
                            )

                            post_id = hashlib.sha256(f"reddit_{subreddit}_{post.get('id', title)}".encode("utf-8")).hexdigest()[:16]

                            results.append(
                                {
                                    "id": post_id,
                                    "title": f"[{subreddit}] {title}",
                                    "url": permalink,
                                    "canonical_url": permalink,
                                    "source": f"Reddit ({subreddit})",
                                    "domain": "reddit.com",
                                    "published_at": now_iso,
                                    "timestamp": now_ts,
                                    "section": "trends",
                                    "region": "kolkata_wb" if "kolkata" in subreddit.lower() else "trends",
                                    "text": body[:1200],
                                    "lead": body[:300],
                                    "is_high_authority": False,
                                    "is_trend": True,
                                }
                            )
                        if results:
                            logger.info(f"Reddit ({subreddit}): Ingested {len(results)} discussions via JSON.")
                            return results
            except Exception as e:
                logger.debug(f"Reddit JSON fetch failed for {subreddit}: {e}")

            # 2. Resilient RSS Fallback if JSON is 403 or fails
            try:
                clean_sub = subreddit.replace("r/", "")
                rss_url = f"https://www.reddit.com/r/{clean_sub}/.rss?sort=top&t=day"
                timeout = aiohttp.ClientTimeout(total=12)
                async with session.get(rss_url, headers=headers, timeout=timeout) as resp:
                    if resp.status == 200:
                        xml_content = await resp.text()
                        feed = feedparser.parse(xml_content)
                        now_iso = datetime.now(timezone.utc).isoformat()
                        now_ts = datetime.now(timezone.utc).timestamp()

                        for entry in feed.entries[:10]:
                            title = entry.get("title", "").strip()
                            link = entry.get("link", "").strip()
                            if not title or not link:
                                continue

                            # Filter out routine social noise: classifieds, buy/sell, megathreads
                            lower_title = title.lower()
                            if any(
                                kw in lower_title
                                for kw in [
                                    "classifieds",
                                    "buy/sell",
                                    "megathread",
                                    "weekly discussion",
                                    "monthly discussion",
                                    "flatmate",
                                    "roommate",
                                ]
                            ):
                                continue

                            summary = entry.get("summary", "") or entry.get("content", [{}])[0].get("value", "")
                            clean_summary = re.sub(r"<[^>]+>", " ", summary).strip()
                            body = f"Community Topic: {title}. {clean_summary[:400]}"

                            post_id = hashlib.sha256(f"reddit_rss_{subreddit}_{title}".encode("utf-8")).hexdigest()[:16]

                            results.append(
                                {
                                    "id": post_id,
                                    "title": f"[{subreddit}] {title}",
                                    "url": link,
                                    "canonical_url": link,
                                    "source": f"Reddit ({subreddit})",
                                    "domain": "reddit.com",
                                    "published_at": now_iso,
                                    "timestamp": now_ts,
                                    "section": "trends",
                                    "region": "kolkata_wb" if "kolkata" in subreddit.lower() else "trends",
                                    "text": body[:1200],
                                    "lead": body[:300],
                                    "is_high_authority": False,
                                    "is_trend": True,
                                }
                            )
                        logger.info(f"Reddit ({subreddit}): Ingested {len(results)} discussions via RSS fallback.")
            except Exception as e:
                logger.warning(f"Error fetching Reddit {subreddit} RSS fallback: {e}")

        return results

    async def fetch_indian_express_trends(
        self, session: aiohttp.ClientSession, url: str
    ) -> List[Dict[str, Any]]:
        """Fetch Indian Express viral trending stories."""
        results: List[Dict[str, Any]] = []
        headers = {"User-Agent": BROWSER_USER_AGENT}

        async with self.semaphore:
            try:
                timeout = aiohttp.ClientTimeout(total=15)
                async with session.get(url, headers=headers, timeout=timeout) as resp:
                    if resp.status != 200:
                        return results
                    xml_content = await resp.text()

                feed = feedparser.parse(xml_content)
                now_iso = datetime.now(timezone.utc).isoformat()
                now_ts = datetime.now(timezone.utc).timestamp()

                for entry in feed.entries[:10]:
                    title = entry.get("title", "").strip()
                    link = entry.get("link", "").strip()
                    summary = entry.get("summary", "") or entry.get("description", "")
                    clean_summary = re.sub(r"<[^>]+>", "", summary).strip()

                    if not title or not link:
                        continue

                    post_id = hashlib.sha256(f"ie_trend_{title}".encode("utf-8")).hexdigest()[:16]

                    results.append(
                        {
                            "id": post_id,
                            "title": title,
                            "url": link,
                            "canonical_url": link,
                            "source": "The Indian Express (Trending)",
                            "domain": "indianexpress.com",
                            "published_at": now_iso,
                            "timestamp": now_ts,
                            "section": "trends",
                            "region": "trends",
                            "text": (clean_summary if len(clean_summary) > 40 else title)[:1200],
                            "lead": clean_summary[:300],
                            "is_high_authority": True,
                            "is_trend": True,
                        }
                    )
            except Exception as e:
                logger.warning(f"Error fetching Indian Express trends: {e}")

        return results

    async def collect_all_trends(self) -> List[Dict[str, Any]]:
        """Run all trends collectors concurrently."""
        all_trends: List[Dict[str, Any]] = []
        connector = aiohttp.TCPConnector(limit=10, ssl=False)

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []

            # 1. Google Trends
            for item in self.trends_cfg.get("google_trends", []):
                tasks.append(self.fetch_google_trends(session, item.get("url", "")))

            # 2. Reddit Subreddits
            for item in self.trends_cfg.get("reddit_subreddits", []):
                tasks.append(
                    self.fetch_reddit_posts(
                        session, item.get("subreddit", "r/reddit"), item.get("url", "")
                    )
                )

            # 3. Indian Express Trends
            for item in self.trends_cfg.get("indian_express_trends", []):
                tasks.append(
                    self.fetch_indian_express_trends(session, item.get("url", ""))
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, list):
                    all_trends.extend(res)

        logger.info(f"Trends collection complete: {len(all_trends)} trend items collected.")
        return all_trends
