"""
Autonomous 24/7 E-Paper News Agent - Trends Subcategory Classifier
Categorizes viral trends into 4 distinct editorial tracks:
1. Internet Buzz (Memes, online debates, viral community discussions)
2. Social Pulse (Civic reactions, public outcry, societal sentiment)
3. Entertainment & Gaming (Creators, cinema, anime, gaming, music)
4. What's Trending (Google Trends search spikes, emerging search topics)
"""

from __future__ import annotations

import re
from typing import Any, Dict

ENTERTAINMENT_KEYWORDS = [
    "movie", "film", "cinema", "trailer", "box office", "actor", "actress", 
    "bollywood", "tollywood", "hollywood", "anime", "manga", "gaming", "game", 
    "esports", "gta", "playstation", "xbox", "song", "album", "concert", "music",
    "singer", "creator", "youtuber", "streamer", "influencer", "celebrity"
]

SOCIAL_PULSE_KEYWORDS = [
    "traffic", "metro", "bus", "transport", "pothole", "civic", "water", 
    "pollution", "air quality", "hospital", "doctor", "strike", "protest", 
    "exam", "student", "teacher", "court", "complaint", "scam", "fraud",
    "police", "safety", "women", "power cut", "load shedding", "rent", "inflation"
]

INTERNET_BUZZ_KEYWORDS = [
    "meme", "viral", "reddit", "x post", "tweet", "post", "thread", "discussion",
    "debate", "opinion", "reaction", "commentary", "trend", "internet", "netizens",
    "online", "community", "social media", "instagram", "reel", "tiktok"
]


def classify_trend_subcategory(story: Dict[str, Any]) -> str:
    """
    Deterministically assign a trend story to one of four structured subcategories.
    """
    headline = (story.get("headline", "") or "").lower()
    text = (story.get("summary", "") or story.get("primary_text", "") or "").lower()
    source = (story.get("source", "") or story.get("domain", "") or "").lower()
    combined = f"{headline} {text} {source}"

    # 1. What's Trending (Search volume spikes from Google Trends)
    if "trends.google" in source or "google trends" in source or "search volume" in text:
        return "What's Trending"

    # 2. Entertainment & Gaming
    if any(kw in combined for kw in ENTERTAINMENT_KEYWORDS):
        return "Entertainment & Gaming"

    # 3. Social Pulse (Civic, public reaction, local issues)
    if any(kw in combined for kw in SOCIAL_PULSE_KEYWORDS):
        return "Social Pulse"

    # 4. Internet Buzz (Reddit, memes, online debates)
    if "reddit" in source or any(kw in combined for kw in INTERNET_BUZZ_KEYWORDS):
        return "Internet Buzz"

    # Default fallback for viral desk
    return "What's Trending"
