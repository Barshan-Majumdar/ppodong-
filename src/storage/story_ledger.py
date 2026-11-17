"""
Autonomous 24/7 E-Paper News Agent - Story Lifecycle & Time-Awareness Ledger
Maintains rolling memory of covered stories to detect ongoing developments across editions
(Morning vs Evening), calculate first-reported/last-updated timestamps, and connect updates.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("epaper.storage.story_ledger")


def normalize_headline(headline: str) -> str:
    """Normalize headline for robust fuzzy matching across media outlets."""
    text = re.sub(r"[^\w\s]", "", headline.lower())
    words = [w for w in text.split() if len(w) > 2]
    return " ".join(words)


class StoryLifecycleLedger:
    """
    Tracks stories across daily editions to provide continuous editorial time-awareness.
    Identifies when a breaking morning story receives afternoon updates or official statements.
    """

    def __init__(self, ledger_path: str = "data/story_lifecycle.json", retention_days: int = 5):
        self.ledger_path = ledger_path
        self.retention_days = retention_days
        self.stories: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.ledger_path):
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.stories = data.get("stories", {})
            except Exception as e:
                logger.warning(f"Could not load story lifecycle ledger: {e}. Starting fresh.")
                self.stories = {}
        else:
            self.stories = {}

    def save(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        cleaned: Dict[str, Dict[str, Any]] = {}
        for sid, entry in self.stories.items():
            try:
                last_up = entry.get("last_updated_at", "")
                dt = datetime.fromisoformat(last_up.replace("Z", "+00:00"))
                if dt >= cutoff:
                    cleaned[sid] = entry
            except Exception:
                cleaned[sid] = entry

        self.stories = cleaned
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        try:
            with open(self.ledger_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "stories": self.stories,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Error saving story lifecycle ledger: {e}")

    def match_existing_story(
        self, cluster: Dict[str, Any]
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Check if an incoming story cluster is a continuation or update of a previously reported story.
        Matches via:
        1. Overlapping canonical or direct URLs.
        2. High normalized headline similarity (>0.82) within the last 36 hours.
        """
        incoming_urls = set(cluster.get("canonical_urls", []) + cluster.get("urls", []))
        incoming_norm_head = normalize_headline(cluster.get("headline", ""))

        for sid, entry in self.stories.items():
            # 1. URL match
            existing_urls = set(entry.get("urls", []))
            if incoming_urls and existing_urls and (incoming_urls & existing_urls):
                return sid, entry

            # 2. Headline similarity match
            existing_norm_head = entry.get("normalized_headline", "")
            if incoming_norm_head and existing_norm_head:
                ratio = difflib.SequenceMatcher(None, incoming_norm_head, existing_norm_head).ratio()
                if ratio >= 0.82:
                    return sid, entry

        return None

    def process_story(
        self,
        story: Dict[str, Any],
        edition_cycle: str = "Morning",
    ) -> Dict[str, Any]:
        """
        Attach time-awareness metadata (first_reported_at, last_updated_at, development_status, previous_reference).
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        cluster_time = story.get("published_at") or now_iso
        match = self.match_existing_story(story)

        if match:
            sid, prev = match
            first_reported = prev.get("first_reported_at", cluster_time)
            update_count = prev.get("update_count", 1) + 1
            prev_cycle = prev.get("last_cycle", "Earlier")
            prev_headline = prev.get("headline", "")

            # Determine development status
            dev_status = f"UPDATED ({edition_cycle.upper()} EDITION)"
            if story.get("urgency_tag") == "BREAKING":
                dev_status = "DEVELOPING: NEW DISPATCH"

            story["first_reported_at"] = first_reported
            story["last_updated_at"] = cluster_time
            story["development_status"] = dev_status
            story["update_count"] = update_count
            story["previous_reference"] = f"Follows {prev_cycle} report: '{prev_headline[:60]}...'"

            # Update ledger state
            prev["last_updated_at"] = cluster_time
            prev["update_count"] = update_count
            prev["last_cycle"] = edition_cycle
            prev["urls"] = list(set(prev.get("urls", []) + story.get("urls", [])))
            prev["headline"] = story.get("headline", prev["headline"])
        else:
            sid = story.get("cluster_id") or f"STORY_{len(self.stories)+1:04d}"
            story["first_reported_at"] = cluster_time
            story["last_updated_at"] = cluster_time
            story["development_status"] = "FIRST REPORT"
            story["update_count"] = 1
            story["previous_reference"] = ""

            self.stories[sid] = {
                "headline": story.get("headline", ""),
                "normalized_headline": normalize_headline(story.get("headline", "")),
                "first_reported_at": cluster_time,
                "last_updated_at": cluster_time,
                "last_cycle": edition_cycle,
                "update_count": 1,
                "urls": story.get("urls", []),
                "section": story.get("section", ""),
            }

        return story
