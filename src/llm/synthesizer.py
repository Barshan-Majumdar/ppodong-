"""
Autonomous 24/7 E-Paper News Agent - Parallel Dual-LLM Map-Reduce Synthesizer
Uses Google Gemini API and Groq API concurrently in parallel across batches of stories,
with mutual failover and structured Pydantic schema validation.
Guarantees 100% story retention and balanced journalistic synthesis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger("epaper.llm.synthesizer")


class StorySummary(BaseModel):
    cluster_id: str = Field(description="The exact cluster_id provided in the input batch")
    headline: str = Field(description="A punchy, professional broadsheet newspaper headline in Times New Roman style")
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
    """
    Parallel Dual-LLM Map-Reduce Synthesizer.
    Executes Gemini API and Groq API concurrently in parallel across partitioned batches,
    merging all outputs into a unified broadsheet edition with zero story omission.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        batch_size: int = 10,
        inter_batch_delay_seconds: float = 2.0,
    ):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.batch_size = batch_size
        self.inter_batch_delay = inter_batch_delay_seconds

        # Lazy clients
        self._gemini_client = None
        self._groq_client = None
        self._gemini_sdk_type = None

    def _get_gemini_client(self):
        if self._gemini_client is None and self.gemini_api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
                self._gemini_sdk_type = "google-genai"
            except ImportError:
                try:
                    import google.generativeai as legacy_genai
                    legacy_genai.configure(api_key=self.gemini_api_key)
                    self._gemini_client = legacy_genai
                    self._gemini_sdk_type = "legacy"
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini legacy client: {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")
        return self._gemini_client

    def _get_groq_client(self):
        if self._groq_client is None and self.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
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
            f"You are the Chief Editor of 'The Daily Chronicle' broadsheet newspaper.\n"
            f"Summarize this batch of EXACTLY {len(batch)} news story clusters.\n\n"
            f"STRICT RULES:\n"
            f"1. Return pure valid JSON with a 'stories' list containing EXACTLY {len(batch)} items.\n"
            f"2. Each item MUST preserve its exact 'cluster_id'. ZERO omission.\n"
            f"3. Write a sharp broadsheet headline, a 2-3 sentence factual journalistic summary, and 1 'key_takeaway'.\n"
            f"4. Assign urgency_tag: 'BREAKING', 'DEVELOPING', 'TRENDING', or 'VERIFIED'.\n\n"
            f"INPUT STORIES ({len(batch)} items):\n"
            f"{json.dumps(input_data, indent=2)}\n\n"
            f"Return JSON schema:\n"
            f'{{"stories": [{{"cluster_id": "...", "headline": "...", "summary": "...", "key_takeaway": "...", "category": "...", "state_or_region": "...", "urgency_tag": "..."}}]}}'
        )
        return prompt

    async def _call_gemini(self, model_name: str, prompt: str) -> Optional[str]:
        """Execute Gemini API call with strict 25s timeout and structured JSON formatting."""
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
                model = client.GenerativeModel(
                    model_name=model_name,
                    generation_config={"response_mime_type": "application/json", "temperature": 0.2},
                )
                response = model.generate_content(prompt)
                return response.text

        try:
            return await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=25.0)
        except asyncio.TimeoutError:
            logger.warning(f"Gemini ({model_name}) timed out after 25s. Switching provider.")
            return None
        except Exception as e:
            logger.warning(f"Gemini ({model_name}) call failed: {e}")
            return None

    async def _call_groq(self, prompt: str) -> Optional[str]:
        """Execute Groq API call across fast open-weight models with 20s timeout."""
        client = self._get_groq_client()
        if not client:
            return None

        groq_models = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]

        def _sync_call(model_id: str):
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior broadsheet newspaper editor. Output pure valid JSON with a 'stories' array.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            return completion.choices[0].message.content

        for model_id in groq_models:
            try:
                resp = await asyncio.wait_for(asyncio.to_thread(_sync_call, model_id), timeout=20.0)
                if resp and len(resp) > 30:
                    return resp
            except Exception as e:
                logger.debug(f"Groq ({model_id}) call failed: {e}")

        return None

    def _parse_and_validate_batch_output(
        self, raw_json: str, batch: List[Dict[str, Any]]
    ) -> Optional[List[StorySummary]]:
        """Validate that all stories in the batch are accounted for."""
        try:
            cleaned = re.sub(r"^```(?:json)?", "", raw_json.strip())
            cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

            parsed = json.loads(cleaned)
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

            if len(output_map) == len(batch):
                return [output_map[c["cluster_id"]] for c in batch]

            # Recover any missing items with extractive fallback so ZERO items are lost
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

    async def summarize_batch_with_provider(
        self,
        batch: List[Dict[str, Any]],
        batch_num: int,
        total_batches: int,
        preferred_provider: str = "gemini",
    ) -> List[StorySummary]:
        """
        Summarizes a single batch with a preferred provider (Gemini or Groq)
        and automatic reciprocal failover to the other provider.
        """
        prompt = self._build_batch_prompt(batch)

        if preferred_provider == "gemini":
            # 1. Try Gemini with fast models first
            if self.gemini_api_key:
                for gm in ["gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"]:
                    logger.info(f"[Parallel Worker: Gemini] Batch {batch_num}/{total_batches}: Querying '{gm}'...")
                    resp = await self._call_gemini(gm, prompt)
                    if resp:
                        res = self._parse_and_validate_batch_output(resp, batch)
                        if res:
                            logger.info(f"[Parallel Worker: Gemini] Batch {batch_num}/{total_batches} finished ({len(res)} stories).")
                            return res

            # 2. Reciprocal Failover to Groq
            if self.groq_api_key:
                logger.info(f"[Failover -> Groq] Batch {batch_num}/{total_batches}: Querying Groq API...")
                resp = await self._call_groq(prompt)
                if resp:
                    res = self._parse_and_validate_batch_output(resp, batch)
                    if res:
                        return res

        else:  # preferred_provider == "groq"
            # 1. Try Groq
            if self.groq_api_key:
                logger.info(f"[Parallel Worker: Groq] Batch {batch_num}/{total_batches}: Querying Groq API...")
                resp = await self._call_groq(prompt)
                if resp:
                    res = self._parse_and_validate_batch_output(resp, batch)
                    if res:
                        logger.info(f"[Parallel Worker: Groq] Batch {batch_num}/{total_batches} finished ({len(res)} stories).")
                        return res

            # 2. Reciprocal Failover to Gemini
            if self.gemini_api_key:
                for gm in ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]:
                    logger.info(f"[Failover -> Gemini] Batch {batch_num}/{total_batches}: Querying '{gm}'...")
                    resp = await self._call_gemini(gm, prompt)
                    if resp:
                        res = self._parse_and_validate_batch_output(resp, batch)
                        if res:
                            return res

        # 3. Deterministic Extractive Fallback
        logger.info(f"Batch {batch_num}/{total_batches}: Utilizing deterministic extractive summarization fallback.")
        return [generate_extractive_fallback_summary(c) for c in batch]

    async def synthesize_all_stories(
        self, ranked_clusters: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Executes Parallel Dual-LLM Synthesis across all batches:
        - Shards batches concurrently across Gemini and Groq
        - Merges outputs into a consolidated broadsheet story collection
        - Retains 100% of unique clusters (zero omission)
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
            f"Starting Parallel Dual-LLM Map-Reduce Synthesis: {total_stories} stories split into "
            f"{total_batches} batches (Gemini & Groq working in parallel)."
        )

        all_summaries: List[StorySummary] = []

        has_both = bool(self.gemini_api_key and self.groq_api_key)

        if has_both and total_batches >= 2:
            # Partition batches into 2 concurrent worker streams
            gemini_batches = [(b, i + 1) for i, b in enumerate(batches) if i % 2 == 0]
            groq_batches = [(b, i + 1) for i, b in enumerate(batches) if i % 2 == 1]

            logger.info(
                f"[PARALLEL DUAL-LLM ACTIVE] "
                f"Gemini Worker assigned {len(gemini_batches)} batches | "
                f"Groq Worker assigned {len(groq_batches)} batches."
            )

            async def _run_gemini_worker():
                results = []
                for b, b_num in gemini_batches:
                    res = await self.summarize_batch_with_provider(
                        b, b_num, total_batches, preferred_provider="gemini"
                    )
                    results.extend(res)
                    await asyncio.sleep(self.inter_batch_delay)
                return results

            async def _run_groq_worker():
                results = []
                for b, b_num in groq_batches:
                    res = await self.summarize_batch_with_provider(
                        b, b_num, total_batches, preferred_provider="groq"
                    )
                    results.extend(res)
                    await asyncio.sleep(self.inter_batch_delay)
                return results

            gemini_res, groq_res = await asyncio.gather(_run_gemini_worker(), _run_groq_worker())
            all_summaries.extend(gemini_res)
            all_summaries.extend(groq_res)

        else:
            # If 1 batch or only 1 key configured
            for idx, batch in enumerate(batches, start=1):
                pref = "gemini" if (idx % 2 == 1 or not self.groq_api_key) else "groq"
                batch_res = await self.summarize_batch_with_provider(
                    batch, idx, total_batches, preferred_provider=pref
                )
                all_summaries.extend(batch_res)
                if idx < total_batches:
                    await asyncio.sleep(self.inter_batch_delay)

        # Merge all summaries back into cluster dictionaries
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
            f"Parallel Synthesis complete: 100% ({len(synthesized_clusters)}/{total_stories}) "
            f"stories synthesized and merged with zero omission."
        )
        return synthesized_clusters
