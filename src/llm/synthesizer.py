"""
Autonomous 24/7 E-Paper News Agent - Zero-Exclusion Batch LLM Synthesizer
Uses Google Gemini API (gemini-3.8-flash primary, gemini-3.5-flash-lite fallback)
and Groq API (llama-3.3-70b-versatile secondary) with Pydantic Structured Outputs.
Processes clusters in batches of 15 to prevent truncation, guaranteeing 100% story retention.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("epaper.llm.synthesizer")


class StorySummary(BaseModel):
    cluster_id: str = Field(description="The exact cluster_id provided in the input batch")
    headline: str = Field(description="A punchy, professional broadsheet newspaper headline")
    summary: str = Field(description="Crisp 2-3 sentence factual journalistic summary")
    key_takeaway: str = Field(description="1 concise bullet point on why this matters or next steps")
    category: str = Field(description="Section category name")
    state_or_region: Optional[str] = Field(default="", description="State or city name if applicable")
    urgency_tag: str = Field(default="VERIFIED", description="BREAKING, DEVELOPING, VERIFIED, or TRENDING")


class BatchSummaryOutput(BaseModel):
    stories: List[StorySummary] = Field(description="List of summarized stories exactly matching the input clusters")


def generate_extractive_fallback_summary(cluster: Dict[str, Any]) -> StorySummary:
    """Deterministic, high-quality fallback summary when LLM APIs are offline or exhausted."""
    headline = cluster.get("headline", "").strip()
    primary_text = cluster.get("primary_text", "").strip()
    lead = cluster.get("lead_summary", "").strip()
    section = cluster.get("section", "National Affairs")
    state_tag = cluster.get("state_tag", "")

    # Clean text of HTML tags
    clean_text = re.sub(r"<[^>]+>", "", primary_text or lead).strip()
    sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    # Pick first 2-3 sentences for summary
    selected_sentences = [s.strip() for s in sentences if len(s.strip()) > 15][:3]
    if not selected_sentences:
        summary_str = f"{headline}. Reports confirm developments unfolding across the region with ongoing monitoring."
    else:
        summary_str = " ".join(selected_sentences)

    sources = ", ".join(cluster.get("sources", [])[:3])
    takeaway = f"Reported by {sources or 'verified news agencies'}; situation remains active."

    tag = "TRENDING" if cluster.get("is_trend") else ("BREAKING" if cluster.get("articles_count", 1) >= 3 else "VERIFIED")

    return StorySummary(
        cluster_id=cluster["cluster_id"],
        headline=headline,
        summary=summary_str[:600],
        key_takeaway=takeaway[:200],
        category=section,
        state_or_region=state_tag or cluster.get("region", ""),
        urgency_tag=tag,
    )


class LLMSynthesizer:
    """Manages multi-tier LLM batch summarization with strict structured output."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        batch_size: int = 15,
        inter_batch_delay_seconds: float = 4.0,
    ):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.batch_size = batch_size
        self.inter_batch_delay = inter_batch_delay_seconds

        # Initialize clients lazily
        self._gemini_client = None
        self._groq_client = None

    def _get_gemini_client(self):
        if self._gemini_client is None and self.gemini_api_key:
            try:
                # First try new google-genai SDK
                from google import genai
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
                self._gemini_sdk_type = "google-genai"
            except ImportError:
                try:
                    # Fallback to google.generativeai
                    import google.generativeai as legacy_genai
                    legacy_genai.configure(api_key=self.gemini_api_key)
                    self._gemini_client = legacy_genai
                    self._gemini_sdk_type = "legacy"
                except ImportError:
                    logger.warning("Neither 'google-genai' nor 'google-generativeai' installed.")
                    self._gemini_client = None
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
                self._gemini_client = None
        return self._gemini_client

    def _get_groq_client(self):
        if self._groq_client is None and self.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key)
            except ImportError:
                logger.warning("'groq' package not installed.")
                self._groq_client = None
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
                self._groq_client = None
        return self._groq_client

    def _build_batch_prompt(self, batch: List[Dict[str, Any]]) -> str:
        """Create prompt instructing the LLM to summarize every single story without dropping any."""
        input_data = []
        for c in batch:
            input_data.append({
                "cluster_id": c["cluster_id"],
                "section": c.get("section", "General"),
                "state_or_region": c.get("state_tag") or c.get("region", ""),
                "headline": c.get("headline", ""),
                "sources": c.get("sources", [])[:4],
                "article_count": c.get("articles_count", 1),
                "text": c.get("primary_text", "")[:1000],
            })

        prompt = (
            f"You are the Chief Editor and Senior Journalist for 'The Daily Chronicle' broadsheet newspaper.\n"
            f"You are given a batch of EXACTLY {len(batch)} news story clusters.\n\n"
            f"STRICT EDITORIAL RULES:\n"
            f"1. You MUST return a JSON object containing a 'stories' array with EXACTLY {len(batch)} items.\n"
            f"2. Each item MUST correspond to one input story and preserve its exact 'cluster_id'.\n"
            f"3. ZERO EXCLUSION: Do NOT merge, skip, drop, or summarize away any cluster_id.\n"
            f"4. For each story, write a punchy, factual broadsheet headline, a concise 2-3 sentence journalistic "
            f"summary, and 1 'key_takeaway' line.\n"
            f"5. Assign urgency_tag: 'BREAKING' (major event or multi-source coverage), 'DEVELOPING', 'TRENDING' "
            f"(viral/social), or 'VERIFIED'.\n\n"
            f"INPUT STORIES ({len(batch)} items):\n"
            f"{json.dumps(input_data, indent=2)}\n\n"
            f"Return ONLY valid JSON matching this schema:\n"
            f'{{"stories": [{{"cluster_id": "...", "headline": "...", "summary": "...", "key_takeaway": "...", "category": "...", "state_or_region": "...", "urgency_tag": "..."}}]}}'
        )
        return prompt

    async def _call_gemini(self, model_name: str, prompt: str) -> Optional[str]:
        """Execute Gemini API call with structured JSON formatting."""
        client = self._get_gemini_client()
        if not client:
            return None

        def _sync_call():
            if self._gemini_sdk_type == "google-genai":
                from google.genai import types
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                )
                return response.text
            else:
                # Legacy SDK
                model = client.GenerativeModel(
                    model_name=model_name,
                    generation_config={"response_mime_type": "application/json", "temperature": 0.2},
                )
                response = model.generate_content(prompt)
                return response.text

        try:
            return await asyncio.to_thread(_sync_call)
        except Exception as e:
            logger.warning(f"Gemini ({model_name}) call failed: {e}")
            return None

    async def _call_groq(self, prompt: str) -> Optional[str]:
        """Execute Groq API call as secondary fallback across candidate models."""
        client = self._get_groq_client()
        if not client:
            return None

        groq_candidates = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

        def _sync_call(model_id: str):
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior newspaper editor. Output pure valid JSON without markdown fences.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            return completion.choices[0].message.content

        for model_id in groq_candidates:
            try:
                resp = await asyncio.to_thread(_sync_call, model_id)
                if resp:
                    return resp
            except Exception as e:
                logger.debug(f"Groq ({model_id}) call failed: {e}")
        return None

    def _parse_and_validate_batch_output(
        self, raw_json: str, batch: List[Dict[str, Any]]
    ) -> Optional[List[StorySummary]]:
        """Validate that all stories in the batch are accounted for."""
        try:
            # Strip markdown fences if present
            cleaned = re.sub(r"^```(?:json)?", "", raw_json.strip())
            cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

            parsed = json.loads(cleaned)
            # Find stories array
            if isinstance(parsed, dict) and "stories" in parsed:
                story_list = parsed["stories"]
            elif isinstance(parsed, list):
                story_list = parsed
            else:
                return None

            expected_ids = {c["cluster_id"] for c in batch}
            output_map: Dict[str, StorySummary] = {}

            for s in story_list:
                cid = s.get("cluster_id")
                if cid and cid in expected_ids:
                    output_map[cid] = StorySummary(
                        cluster_id=cid,
                        headline=s.get("headline", ""),
                        summary=s.get("summary", ""),
                        key_takeaway=s.get("key_takeaway", ""),
                        category=s.get("category", ""),
                        state_or_region=s.get("state_or_region", ""),
                        urgency_tag=s.get("urgency_tag", "VERIFIED"),
                    )

            # If every cluster in the batch was successfully summarized:
            if len(output_map) == len(batch):
                return [output_map[c["cluster_id"]] for c in batch]

            logger.warning(
                f"Batch validation partial mismatch: got {len(output_map)} of {len(batch)}. "
                f"Recovering missing items via extractive fallback."
            )
            # Fill missing items so ZERO items are ever dropped
            complete_list: List[StorySummary] = []
            for c in batch:
                cid = c["cluster_id"]
                if cid in output_map:
                    complete_list.append(output_map[cid])
                else:
                    complete_list.append(generate_extractive_fallback_summary(c))
            return complete_list
        except Exception as e:
            logger.warning(f"Failed to parse LLM structured JSON output: {e}")
            return None

    async def summarize_batch(
        self, batch: List[Dict[str, Any]], batch_num: int, total_batches: int
    ) -> List[StorySummary]:
        """
        Summarize a single batch of clusters through multi-tier fallback:
        Tier 1: Gemini 3.8 Flash / 3.5 Flash / 3.5 Flash-Lite
        Tier 2: Groq (Llama-3.3 / GPT-OSS / Qwen)
        Tier 3: High-Quality Extractive Fallback (Guarantees 100% completion)
        """
        prompt = self._build_batch_prompt(batch)

        # Tier 1: Gemini API
        if self.gemini_api_key:
            gemini_candidates = [
                "gemini-3.8-flash",
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite",
                "gemini-flash-latest",
            ]
            for g_model in gemini_candidates:
                logger.info(f"Batch {batch_num}/{total_batches}: Querying Gemini model '{g_model}'...")
                resp = await self._call_gemini(g_model, prompt)
                if resp:
                    validated = self._parse_and_validate_batch_output(resp, batch)
                    if validated:
                        return validated

        # Tier 2: Groq API
        if self.groq_api_key:
            logger.info(f"Batch {batch_num}/{total_batches}: Querying Groq fallback models...")
            resp = await self._call_groq(prompt)
            if resp:
                validated = self._parse_and_validate_batch_output(resp, batch)
                if validated:
                    return validated

        # Tier 3: Deterministic Extractive fallback
        logger.info(f"Batch {batch_num}/{total_batches}: Utilizing deterministic extractive summarization fallback.")
        return [generate_extractive_fallback_summary(c) for c in batch]

    async def synthesize_all_stories(
        self, ranked_clusters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Processes 100% of ranked clusters across batches with zero exclusion.
        Merges generated journalistic summaries back into cluster dictionaries.
        """
        if not ranked_clusters:
            return []

        total_stories = len(ranked_clusters)
        batches = [
            ranked_clusters[i : i + self.batch_size]
            for i in range(0, total_stories, self.batch_size)
        ]
        total_batches = len(batches)

        logger.info(
            f"Starting Zero-Exclusion Batch LLM Synthesis: {total_stories} stories split into "
            f"{total_batches} batches of up to {self.batch_size} clusters each."
        )

        all_summaries: List[StorySummary] = []

        for idx, batch in enumerate(batches, start=1):
            batch_result = await self.summarize_batch(batch, idx, total_batches)
            all_summaries.extend(batch_result)

            # Polite delay between batches to respect free tier RPM limits if using LLM
            if idx < total_batches and (self.gemini_api_key or self.groq_api_key):
                await asyncio.sleep(self.inter_batch_delay)

        # Map summaries back to the cluster dicts
        summary_by_id = {s.cluster_id: s for s in all_summaries}
        synthesized_clusters: List[Dict[str, Any]] = []

        for cluster in ranked_clusters:
            cid = cluster["cluster_id"]
            story = summary_by_id.get(cid)
            if not story:
                story = generate_extractive_fallback_summary(cluster)

            cluster_copy = dict(cluster)
            cluster_copy["headline"] = story.headline or cluster.get("headline", "")
            cluster_copy["summary"] = story.summary
            cluster_copy["key_takeaway"] = story.key_takeaway
            cluster_copy["urgency_tag"] = story.urgency_tag
            cluster_copy["category"] = story.category or cluster.get("section", "")
            cluster_copy["state_or_region"] = story.state_or_region or cluster.get("state_tag", "")

            synthesized_clusters.append(cluster_copy)

        logger.info(
            f"Synthesis complete: 100% ({len(synthesized_clusters)}/{total_stories}) "
            f"stories synthesized with zero omission."
        )
        return synthesized_clusters
