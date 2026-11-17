import tempfile
import pytest
from src.publisher.verification import classify_story_verification
from src.publisher.trends_classifier import classify_trend_subcategory
from src.publisher.quality_gate import EditorialQualityGate
from src.storage.story_ledger import StoryLifecycleLedger
from src.publisher.epaper_builder import EPaperBuilder


def test_classify_story_verification_official():
    story = {
        "sources": ["Press Information Bureau", "PIB Kolkata"],
        "domains": ["pib.gov.in"],
        "urls": ["https://pib.gov.in/PressReleasePage.aspx?PRID=12345"],
        "headline": "Cabinet approves strategic railway infrastructure expansion in Eastern India",
        "primary_text": "The Union Cabinet has sanctioned railway electrification and double track projects.",
        "urgency_tag": "VERIFIED",
    }
    assert classify_story_verification(story) == "OFFICIAL UPDATE"


def test_classify_story_verification_cross_verified():
    story = {
        "sources": ["The Hindu", "Indian Express", "Reuters"],
        "domains": ["thehindu.com", "indianexpress.com", "reuters.com"],
        "urls": ["https://thehindu.com/story1", "https://indianexpress.com/story1"],
        "headline": "RBI maintains repo rate at 6.5 percent amid inflation moderation",
        "primary_text": "The Monetary Policy Committee voted unanimously to keep rates steady.",
        "articles_count": 3,
        "urgency_tag": "VERIFIED",
    }
    assert classify_story_verification(story) == "CROSS-VERIFIED"


def test_classify_story_verification_developing():
    story = {
        "sources": ["ANI"],
        "domains": ["aninews.in"],
        "urls": ["https://aninews.in/breaking"],
        "headline": "Search underway after passenger vessel experiences engine failure in Hooghly",
        "primary_text": "Rescue operations are currently underway as disaster management personnel respond.",
        "articles_count": 1,
        "urgency_tag": "BREAKING",
    }
    assert classify_story_verification(story) == "DEVELOPING"


def test_classify_trend_subcategories():
    buzz_story = {
        "headline": "Viral meme format sparks humorous commentary across online communities",
        "summary": "Netizens share funny adaptations of everyday social situations on Reddit.",
        "source": "Reddit (r/kolkata)",
    }
    assert classify_trend_subcategory(buzz_story) == "Internet Buzz"

    pulse_story = {
        "headline": "Citizens voice frustration over recurring EM Bypass waterlogging after brief shower",
        "summary": "Commuters face traffic slowdowns as civic authorities inspect drainage networks.",
        "source": "Times of India",
    }
    assert classify_trend_subcategory(pulse_story) == "Social Pulse"

    gaming_story = {
        "headline": "GTA 6 release countdown triggers viral online community theories",
        "summary": "Gaming creators analyze leaked development roadmap screenshots.",
        "source": "Gaming Portal",
    }
    assert classify_trend_subcategory(gaming_story) == "Entertainment & Gaming"

    trending_story = {
        "headline": "Search volume surges for WBJEE counselling dates",
        "summary": "Google Trends records spike in student queries across West Bengal.",
        "source": "Google Trends",
    }
    assert classify_trend_subcategory(trending_story) == "What's Trending"


def test_quality_gate_audit_and_duplicate_purge():
    gate = EditorialQualityGate()
    front_leads = [
        {
            "cluster_id": "STORY_01",
            "headline": "Kolkata Port handles record cargo volumes in fiscal year review",
            "summary": "Syama Prasad Mookerjee Port recorded substantial growth in container movements.",
            "key_takeaway": "Port modernization initiatives reduced vessel turnaround time by 18 hours.",
            "sources": ["The Hindu"],
            "section": "kolkata_bengal",
        }
    ]
    sections_data = {
        "kolkata_bengal": [
            {
                # Duplicate that slipped in:
                "cluster_id": "STORY_01",
                "headline": "Kolkata Port handles record cargo volumes in fiscal year review",
                "summary": "Syama Prasad Mookerjee Port recorded substantial growth in container movements.",
                "key_takeaway": "Port modernization initiatives reduced vessel turnaround time by 18 hours.",
                "sources": ["The Hindu"],
                "section": "kolkata_bengal",
            },
            {
                "cluster_id": "STORY_02",
                "headline": "Alipore Zoo introduces new climate-controlled aviary enclosures",
                "summary": "Veterinary experts oversee the safe relocation of rare Himalayan pheasants.",
                "key_takeaway": "The upgrades provide improved winter housing for endangered migratory birds.",
                "sources": ["Telegraph India"],
                "section": "kolkata_bengal",
            },
        ],
        "indian_states": [],
        "national": [],
        "global": [],
        "trends": [],
    }

    all_passed, checklist_md, clean_front, clean_sections = gate.run_pre_publication_audit(
        front_leads, sections_data
    )

    # Assert duplicate was purged from section data!
    sec_ids = [s["cluster_id"] for s in clean_sections["kolkata_bengal"]]
    assert "STORY_01" not in sec_ids
    assert "STORY_02" in sec_ids
    assert "Pre-publication Editorial Quality Audit" in checklist_md


def test_story_lifecycle_ledger():
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = f"{tmpdir}/story_lifecycle.json"
        ledger = StoryLifecycleLedger(ledger_path=ledger_path)

        story_morning = {
            "cluster_id": "STORY_TRAM_01",
            "headline": "Heritage Tram route preservation proposal submitted to Transport Department",
            "urls": ["https://example.com/tram-news-1"],
            "published_at": "2026-11-17T06:30:00+00:00",
            "section": "kolkata_bengal",
        }
        processed_morning = ledger.process_story(story_morning, edition_cycle="Morning")
        assert processed_morning["development_status"] == "FIRST REPORT"
        assert processed_morning["update_count"] == 1
        ledger.save()

        # Evening update on same topic
        story_evening = {
            "cluster_id": "STORY_TRAM_02",
            "headline": "Heritage Tram route preservation proposal approved by Transport Minister",
            "urls": ["https://example.com/tram-news-1", "https://example.com/tram-minister-statement"],
            "published_at": "2026-11-17T16:30:00+00:00",
            "section": "kolkata_bengal",
        }
        processed_evening = ledger.process_story(story_evening, edition_cycle="Evening")
        assert "UPDATED" in processed_evening["development_status"]
        assert processed_evening["update_count"] == 2
        assert processed_evening["first_reported_at"] == "2026-11-17T06:30:00+00:00"
