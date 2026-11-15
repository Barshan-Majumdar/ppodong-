from src.publisher.distributor import Distributor


def test_distributor_caption_formatting():
    distributor = Distributor()
    stories = [
        {
            "headline": "Kolkata Green Hydrogen Plant inaugurated",
            "state_or_region": "KOLKATA",
            "category": "Kolkata & West Bengal",
        },
        {
            "headline": "RBI maintains repo rate in monetary policy announcement",
            "state_or_region": "INDIA",
            "category": "National",
        },
    ]

    caption = distributor.format_telegram_caption(
        edition_date="2026-11-15",
        total_stories=25,
        total_sources=8,
        stories=stories,
    )

    assert "THE DAILY CHRONICLE" in caption
    assert "2026-11-15" in caption
    assert "25 Verified Stories" in caption
    assert "Green Hydrogen" in caption


def test_distributor_skips_when_no_credentials():
    distributor = Distributor(telegram_token="", telegram_chat_id="")
    ok = distributor.send_to_telegram(
        pdf_path=None,
        md_path=None,
        edition_date="2026-11-15",
        total_stories=10,
        total_sources=3,
        stories=[],
    )
    assert ok is False
