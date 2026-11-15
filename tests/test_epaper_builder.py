import os
import tempfile
from src.publisher.epaper_builder import EPaperBuilder


def test_epaper_builder_html_and_markdown():
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = EPaperBuilder(templates_dir="templates", editions_dir=tmpdir)

        sample_stories = [
            {
                "cluster_id": "CLUST_01",
                "headline": "Kolkata Book Fair sets record international pavilion attendance",
                "summary": "The 48th International Kolkata Book Fair saw over 2 million footfalls over the weekend.",
                "key_takeaway": "Cultural tourism and publishing sales registered a 14% year-on-year increase.",
                "category": "Kolkata & West Bengal Desk",
                "state_or_region": "KOLKATA",
                "section": "kolkata_bengal",
                "urgency_tag": "VERIFIED",
                "sources": ["The Hindu", "Telegraph India"],
                "articles_count": 2,
                "urls": ["https://example.com/story-1"],
            },
            {
                "cluster_id": "CLUST_02",
                "headline": "India Semiconductor Mission approves three new fabrication units",
                "summary": "Cabinet sanction provides greenlight for advanced chip packaging units in Gujarat and Assam.",
                "key_takeaway": "Commercial production anticipated by mid-2027.",
                "category": "National Affairs, Governance & Economy",
                "state_or_region": "INDIA NATIONAL",
                "section": "national",
                "urgency_tag": "BREAKING",
                "sources": ["PIB", "Indian Express", "Reuters"],
                "articles_count": 3,
                "urls": ["https://example.com/story-2"],
            },
        ]

        edition_date = "2026-11-15"
        res = builder.publish_edition(sample_stories, edition_date=edition_date)

        assert res["date"] == edition_date
        assert res["total_stories"] == 2
        assert os.path.exists(res["html_path"])
        assert os.path.exists(res["md_path"])

        # Check HTML content
        with open(res["html_path"], "r", encoding="utf-8") as f:
            html = f.read()
            assert "The Daily Chronicle" in html
            assert "Kolkata Book Fair" in html
            assert "Semiconductor Mission" in html

        # Check Markdown content
        with open(res["md_path"], "r", encoding="utf-8") as f:
            md = f.read()
            assert "# THE DAILY CHRONICLE" in md
            assert "Kolkata Book Fair" in md
            assert "Semiconductor Mission" in md
            assert "Edition Stats:" in md
