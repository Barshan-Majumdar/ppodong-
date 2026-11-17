"""
Autonomous 24/7 E-Paper News Agent - Source Credibility & Verification Classifier
Calculates deterministic verification confidence for every news story.
Replaces decorative badges with meaningful editorial verification tiers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set
from urllib.parse import urlparse

OFFICIAL_DOMAINS = {
    "pib.gov.in",
    "gov.in",
    "nic.in",
    "mea.gov.in",
    "isro.gov.in",
    "rbi.org.in",
    "eci.gov.in",
    "mohfw.gov.in",
    "wbsedcl.in",
    "kolkatapolice.gov.in",
    "sci.gov.in",
    "delhipolice.gov.in",
}

OFFICIAL_KEYWORDS = [
    "pib",
    "press information bureau",
    "government of india",
    "official gazette",
    "supreme court",
    "high court",
    "ministry of",
    "department of",
    "police",
    "spokesperson",
    "isro",
    "rbi",
    "commissioner",
]

DEVELOPING_KEYWORDS = [
    "death toll rises",
    "unfolding",
    "initial report",
    "rescue underway",
    "search underway",
    "investigation launched",
    "probe ordered",
    "developing story",
    "shots fired",
    "breaking",
]


def classify_story_verification(story: Dict[str, Any]) -> str:
    """
    Deterministically assign one of four verification tiers:
    - OFFICIAL UPDATE: Institutional/Govt primary announcement
    - CROSS-VERIFIED: Independently confirmed across multiple outlets
    - DEVELOPING: Unfolding situation with active updates
    - SINGLE-SOURCE: Single publisher report
    """
    sources = story.get("sources", [])
    domains = story.get("domains", [])
    raw_urls = story.get("urls", [])
    headline = (story.get("headline", "") or "").lower()
    text = (story.get("primary_text", "") or story.get("summary", "") or "").lower()

    # Extract all domains from URLs if not populated
    all_domains: Set[str] = set()
    for d in domains:
        if d:
            all_domains.add(d.lower())
    for u in raw_urls:
        try:
            parsed = urlparse(u).netloc.lower()
            if parsed.startswith("www."):
                parsed = parsed[4:]
            if parsed:
                all_domains.add(parsed)
        except Exception:
            pass

    # 1. Official Update Check
    for d in all_domains:
        if any(d.endswith(off) or off in d for off in OFFICIAL_DOMAINS):
            return "OFFICIAL UPDATE"

    for src in sources:
        src_lower = src.lower()
        if any(kw in src_lower for kw in OFFICIAL_KEYWORDS):
            return "OFFICIAL UPDATE"

    # 2. Developing Story Check
    is_breaking_tag = story.get("urgency_tag") in ("BREAKING", "DEVELOPING")
    has_developing_keywords = any(kw in headline or kw in text for kw in DEVELOPING_KEYWORDS)
    if is_breaking_tag or has_developing_keywords:
        # If it's single source and fresh, it's developing
        return "DEVELOPING"

    # 3. Cross-Verified Check
    distinct_credible_sources = len(set(sources))
    distinct_domains = len(all_domains)
    articles_count = story.get("articles_count", 1)

    if distinct_credible_sources >= 2 or distinct_domains >= 2 or articles_count >= 2:
        return "CROSS-VERIFIED"

    # 4. Single-Source Default
    return "SINGLE-SOURCE"
