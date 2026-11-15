import math
import time
from datetime import datetime, timezone
from src.ml.deduplicator_ranker import DeduplicatorRanker


def test_deduplicator_scoring_and_zero_exclusion():
    config = {
        "pipeline": {
            "deduplication_threshold": 0.80,
            "rolling_window_hours": 26,
        },
        "weights": {
            "coverage_weight": 3.0,
            "regional_boost": {
                "kolkata_wb": 2.5,
                "indian_states": 1.5,
                "national": 1.5,
                "global": 1.0,
                "trends": 1.0,
            },
            "source_authority": {
                "high": 1.5,
                "standard": 1.0,
            },
            "recency_max_bonus": 2.0,
        },
        "authority_domains": {
            "high": ["thehindu.com", "indianexpress.com"]
        },
    }

    ranker = DeduplicatorRanker(config)
    now_ts = datetime.now(timezone.utc).timestamp()

    # 3 duplicate articles on Kolkata Metro from different outlets
    articles = [
        {
            "id": "art_1",
            "title": "Kolkata Metro completes underwater tunnel trials across Hooghly river",
            "lead": "Kolkata Metro Railway achieved a historic milestone as trains ran underwater.",
            "url": "https://thehindu.com/metro-1",
            "canonical_url": "https://thehindu.com/metro-1",
            "source": "The Hindu",
            "domain": "thehindu.com",
            "published_at": datetime.now(timezone.utc).isoformat(),
            "timestamp": now_ts - 1800, # 30 min ago
            "section": "kolkata_bengal",
            "region": "kolkata_wb",
            "text": "Full article on Kolkata underwater metro tunnel trial run.",
            "is_high_authority": True,
        },
        {
            "id": "art_2",
            "title": "Kolkata Metro conducts underwater trials under Hooghly river",
            "lead": "Underwater metro in Kolkata crossed another barrier under river Hooghly.",
            "url": "https://timesofindia.indiatimes.com/metro-2",
            "canonical_url": "https://timesofindia.indiatimes.com/metro-2",
            "source": "Times of India",
            "domain": "indiatimes.com",
            "published_at": datetime.now(timezone.utc).isoformat(),
            "timestamp": now_ts - 2400,
            "section": "kolkata_bengal",
            "region": "kolkata_wb",
            "text": "TOI report on underwater metro trials in Kolkata under Hooghly.",
            "is_high_authority": False,
        },
        # 1 distinct article on ISRO Chandrayaan mission
        {
            "id": "art_3",
            "title": "ISRO announces next lunar exploration timeline for Chandrayaan series",
            "lead": "Indian Space Research Organisation confirmed plans for Chandrayaan-4.",
            "url": "https://indianexpress.com/isro-1",
            "canonical_url": "https://indianexpress.com/isro-1",
            "source": "Indian Express",
            "domain": "indianexpress.com",
            "published_at": datetime.now(timezone.utc).isoformat(),
            "timestamp": now_ts - 3600,
            "section": "national",
            "region": "national",
            "text": "Detailed statement from ISRO Chairman on Chandrayaan-4 mission goals.",
            "is_high_authority": True,
        },
    ]

    ranked = ranker.deduplicate_and_rank(articles)

    # ZERO NEWS EXCLUSION: All topics must be represented
    assert len(ranked) >= 2, "Expected at least 2 distinct clusters"

    # Find the Kolkata Metro cluster
    kolkata_cluster = next((c for c in ranked if "Metro" in c["headline"]), None)
    assert kolkata_cluster is not None
    assert kolkata_cluster["articles_count"] >= 1
    assert "The Hindu" in kolkata_cluster["sources"]

    # Verify score calculation formula
    # Score = W_coverage * ln(1 + |C|) + W_region + W_auth + W_recency
    expected_coverage = 3.0 * math.log(1 + kolkata_cluster["articles_count"])
    assert kolkata_cluster["score"] > expected_coverage
