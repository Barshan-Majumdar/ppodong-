import os
import json
import tempfile
from datetime import datetime, timedelta, timezone
from src.collectors.rss_collector import (
    extract_domain,
    decode_google_news_url_fallback,
    SeenUrlLedger,
    RSSCollector,
)


def test_extract_domain():
    assert extract_domain("https://www.thehindu.com/news/national/") == "thehindu.com"
    assert extract_domain("http://indianexpress.com/article/123") == "indianexpress.com"
    assert extract_domain("https://bengali.abplive.com/news/") == "bengali.abplive.com"
    assert extract_domain("invalid-url") == ""


def test_decode_google_news_url_fallback():
    # If not a google news url, returns None
    assert decode_google_news_url_fallback("https://thehindu.com/news/1") is None


def test_seen_url_ledger():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_file = os.path.join(tmpdir, "seen_test.json")
        ledger = SeenUrlLedger(ledger_path=ledger_file, retention_days=3)

        assert not ledger.is_seen("https://example.com/story-1")
        ledger.mark_seen("https://example.com/story-1")
        assert ledger.is_seen("https://example.com/story-1")

        # Save and reload
        ledger.save()
        assert os.path.exists(ledger_file)

        new_ledger = SeenUrlLedger(ledger_path=ledger_file, retention_days=3)
        assert new_ledger.is_seen("https://example.com/story-1")
        assert not new_ledger.is_seen("https://example.com/story-2")


def test_rss_collector_parse_time():
    config = {
        "pipeline": {"rolling_window_hours": 26},
        "authority_domains": {"high": ["thehindu.com"]},
    }
    collector = RSSCollector(config)
    now = datetime.now(timezone.utc)

    # Empty entry returns now
    t = collector.parse_entry_time({}, now)
    assert t == now
