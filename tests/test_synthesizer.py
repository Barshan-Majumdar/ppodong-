import pytest
from src.llm.synthesizer import (
    LLMSynthesizer,
    StorySummary,
    generate_extractive_fallback_summary,
)


def test_extractive_fallback_summary():
    cluster = {
        "cluster_id": "CLUST_0001_abc12345",
        "headline": "Bengal Global Business Summit announced for November in Kolkata",
        "primary_text": "The West Bengal government has announced the upcoming dates for the Bengal Global Business Summit. Key industrial leaders and international delegates are expected to attend the summit in Kolkata. Infrastructure and technology sectors will be highlighted.",
        "lead_summary": "West Bengal announces dates for Bengal Global Business Summit.",
        "section": "kolkata_bengal",
        "region": "kolkata_wb",
        "state_tag": "WEST BENGAL",
        "sources": ["The Hindu", "Telegraph India"],
        "articles_count": 2,
    }

    summary = generate_extractive_fallback_summary(cluster)
    assert summary.cluster_id == "CLUST_0001_abc12345"
    assert "Bengal Global Business Summit" in summary.headline
    assert len(summary.summary) > 20
    assert summary.key_takeaway != ""
    assert summary.urgency_tag in ["BREAKING", "DEVELOPING", "VERIFIED", "TRENDING"]


@pytest.mark.asyncio
async def test_batch_synthesizer_zero_exclusion():
    synthesizer = LLMSynthesizer(batch_size=2)
    clusters = [
        {
            "cluster_id": "C_1",
            "headline": "Historic Kolkata Tram service preservation plan tabled",
            "lead_summary": "Citizens advocate for heritage route retention.",
            "primary_text": "A plan to modernize and retain heritage tram routes in Kolkata was tabled today.",
            "section": "kolkata_bengal",
            "region": "kolkata_wb",
            "sources": ["Times of India"],
            "articles_count": 1,
        },
        {
            "cluster_id": "C_2",
            "headline": "National Highway expansion across eastern corridor approved",
            "lead_summary": "Cabinet approves key infrastructure project.",
            "primary_text": "The Union Cabinet approved a multilane highway expansion spanning four states.",
            "section": "national",
            "region": "national",
            "sources": ["PIB"],
            "articles_count": 1,
        },
    ]

    # Synthesize without API keys triggers the extractive fallback tier
    results = await synthesizer.synthesize_all_stories(clusters)
    assert len(results) == len(clusters), "Synthesizer must retain 100% of input clusters"
    assert results[0]["cluster_id"] == "C_1"
    assert results[1]["cluster_id"] == "C_2"
    assert "summary" in results[0]
    assert "key_takeaway" in results[0]
