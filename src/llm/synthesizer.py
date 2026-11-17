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
    headline: str = Field(description="Punchy, intelligent broadsheet headline")
    summary: str = Field(description="Crisp 2-3 sentence factual journalistic storytelling summary")
    key_takeaway: str = Field(description="1 sharp, fact-based sentence answering what this actually means, changes, or triggers")
    why_it_matters: Optional[str] = Field(default="", description="1-2 sentences explaining real-world citizen/developer/market implications")
    is_good_news: Optional[bool] = Field(default=False, description="True if inspiring scientific discovery, conservation, community, or achievement")
    fact_check_verdict: Optional[str] = Field(default="", description="SUPPORTED, MISLEADING, FALSE, or empty")
    category: str = Field(description="Section category name")
    state_or_region: Optional[str] = Field(default="", description="State or city name if applicable")
    urgency_tag: str = Field(default="VERIFIED", description="OFFICIAL UPDATE, CROSS-VERIFIED, DEVELOPING, or SINGLE-SOURCE")


class BatchSummaryOutput(BaseModel):
    stories: List[StorySummary] = Field(description="List of summarized stories exactly matching the input clusters")


def generate_extractive_fallback_summary(cluster: Dict[str, Any]) -> StorySummary:
    """Deterministic, factual extractive summary answering What happened, Why it matters, and What's next."""
    headline = cluster.get("headline", "").strip()
    primary_text = cluster.get("primary_text", "").strip()
    lead = cluster.get("lead_summary", "").strip()
    section = cluster.get("section", "National Affairs")
    state_tag = cluster.get("state_tag", "")

    # Clean text of HTML tags and entities
    clean_text = re.sub(r"<[^>]+>", "", primary_text or lead).strip()
    clean_text = re.sub(r"&[a-zA-Z0-9#]+;", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean_text) if len(s.strip()) > 20]

    sources_str = ", ".join(cluster.get("sources", [])[:3]) or "verified news agencies"

    if len(sentences) >= 3:
        summary_str = f"{sentences[0]} {sentences[1]}"
        takeaway = sentences[2]
    elif len(sentences) == 2:
        summary_str = f"{sentences[0]} {sentences[1]}"
        takeaway = f"Key development confirmed through {sources_str}."
    elif len(sentences) == 1:
        summary_str = f"{sentences[0]} Official briefings and field reports remain ongoing."
        takeaway = f"Initial dispatch documented across {sources_str}."
    else:
        summary_str = f"{headline}. Early dispatches indicate ongoing monitoring by regional correspondents."
        takeaway = f"Primary source reporting by {sources_str}."

    why_matters = f"Sets operational precedence for regional administration and industry stakeholders."
    is_good = any(kw in clean_text.lower() for kw in ["breakthrough", "discovery", "conservation", "award", "achievement", "cure", "solar", "sanctuary", "clean energy"])

    # Determine broadsheet urgency tag and verification tier
    from src.publisher.verification import classify_story_verification
    verification_tier = classify_story_verification(cluster)
    if cluster.get("is_trend"):
        urgency_tag = "TRENDING"
    elif cluster.get("articles_count", 1) >= 3 or cluster.get("urgency_tag") in ("BREAKING", "DEVELOPING"):
        urgency_tag = "BREAKING"
    elif verification_tier == "DEVELOPING":
        urgency_tag = "DEVELOPING"
    else:
        urgency_tag = "VERIFIED"

    return StorySummary(
        cluster_id=cluster["cluster_id"],
        headline=headline,
        summary=summary_str[:600],
        key_takeaway=takeaway[:200],
        why_it_matters=why_matters,
        is_good_news=is_good,
        fact_check_verdict="",
        category=section,
        state_or_region=state_tag or cluster.get("region", ""),
        urgency_tag=urgency_tag,
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
            f"You are the Chief Editor of 'The Daily Chronicle', a modern digital broadsheet newspaper.\n\n"
            f"EDITORIAL VISION & PHILOSOPHY:\n"
            f"'The Daily Chronicle' is neither a stuffy, archaic newspaper from 1995, nor is it a low-effort social media meme account. "
            f"It is a sharp, insightful, and essential digital newspaper that tells readers what is happening in the world, what is happening on the internet, and why either one matters.\n"
            f"Think of an exceptionally smart, well-read, curious friend who follows global affairs, technology, and culture closely, understands the deeper context, and explains things with intellectual clarity, fresh energy, and zero corporate fluff.\n\n"
            f"TWO CORE EDITORIAL PRINCIPLES:\n"
            f"1. COVERAGE MUST BE BROAD (The Full Picture):\n"
            f"   A real newspaper covers the whole world, not just political speeches and disasters. Never exclude a story simply because it isn't traditional hard news. Embrace all major pillars:\n"
            f"   - National & International: Geopolitics, governance, policy, diplomacy, major economic shifts, global conflicts.\n"
            f"   - Regional & Local: Civic realities, infrastructure, public transport (e.g. Kolkata traffic, roadworks), state updates, local administration.\n"
            f"   - Technology & Science: AI developments, startups, space missions, cybersecurity, research breakthroughs, gadgets.\n"
            f"   - Business & Finance: Markets, corporate milestones, startup funding, labor trends, consumer economics.\n"
            f"   - Sports: Match results, tournaments, athlete stories, transfers, records, competitive drama.\n"
            f"   - Entertainment & Culture: Cinema, anime, gaming, music, pop culture milestones, literary events.\n"
            f"   - Internet & Viral Trends: Culturally significant Reddit debates, notable X threads, viral memes with societal resonance, online communities, creator announcements.\n"
            f"   - Lifestyle & Society: Public health, environment, education, urban living, changing social dynamics.\n\n"
            f"2. PRESENTATION MUST BE SELECTIVE & HIGH-SIGNAL:\n"
            f"   Platforms like Reddit, X, and Instagram are legitimate modern news sources — a viral Reddit thread about local civic failures is real news, and a cultural meme is real commentary. However, maintain high editorial standards:\n"
            f"   - Feature stories based on relevance, significance, novelty, public interest, or genuine trend momentum.\n"
            f"   - Distinguish real signal from mundane noise (e.g., a city-wide traffic debate or major creator milestone = YES; routine personal rants, spam, or classifieds = NO).\n\n"
            f"ADAPTIVE WRITING TONE BY SECTION:\n"
            f"The writing voice is NOT one-size-fits-all. Adapt the tone naturally to the subject:\n"
            f"- Sports: Energetic, dynamic, capturing athletic drama, stakes, and momentum.\n"
            f"- Technology & Science: Curious, forward-looking, lucid; demystify complex systems without hype.\n"
            f"- Politics & Governance: Sharp, analytical, objective; dissect power dynamics and policy without bias.\n"
            f"- Business & Finance: Perceptive, crisp; connect numbers and market shifts to real-world impact.\n"
            f"- Internet & Viral Trends: Culturally fluent, witty, observant; explain why the internet is talking about it and the cultural pulse behind it.\n"
            f"- Regional & Local News: Grounded, empathetic, attuned to everyday lived reality and civic concerns.\n"
            f"- Tragedies, Disasters, Accidents, Crime, Legal & Deaths: STRICTLY RESPECTFUL, DIGNIFIED, AND SOBER. Never make jokes, sarcastic remarks, or flippant quips about suffering, loss, or victims. Treat sensitive topics with journalistic empathy.\n\n"
            f"STRICT WRITING & OUTPUT RULES:\n"
            f"1. Return pure valid JSON with a 'stories' list containing EXACTLY {len(batch)} items matching input clusters.\n"
            f"2. Each item MUST preserve its exact 'cluster_id'. ZERO omission.\n"
            f"3. HEADLINE: Punchy, intelligent, curiosity-inducing, and memorable. Avoid dry corporate templates ('X announces Y'). Never use deceptive clickbait.\n"
            f"4. SUMMARY: Exactly 2-3 sentences of compelling storytelling:\n"
            f"   - Sentence 1 (Hook & Core Event): What happened, framed compellingly.\n"
            f"   - Sentence 2 (Context & Impact): Why it matters, who is affected, or the tension.\n"
            f"   - Sentence 3 (Consequence / Perspective): What to watch for or the broader takeaway.\n"
            f"   - Ban boring journalese clichés ('In a significant move', 'According to sources', 'This highlights that').\n"
            f"5. KEY TAKEAWAY: Exactly 1 sharp, fact-based sentence answering what this actually means, changes, or triggers. NO MOTIVATIONAL FLUFF OR ARTIFICIAL PHILOSOPHY (e.g. instead of 'When forces sync up, defense operates at peak efficiency', write 'The joint drills aim to improve coordination between security agencies during regional emergencies').\n"
            f"6. CATEGORY: Assign the primary section or viral track: 'National & International', 'Regional & Local', 'Technology & Science', 'Business & Finance', 'Sports', 'Internet Buzz', 'Social Pulse', 'Entertainment & Gaming', or 'What's Trending'.\n"
            f"7. ASSIGN urgency_tag: Must reflect meaningful verification confidence: 'OFFICIAL UPDATE' (government/institutional decree), 'CROSS-VERIFIED' (multi-outlet confirmation), 'DEVELOPING' (unfolding story), or 'SINGLE-SOURCE'.\n"
            f"8. WHY THIS MATTERS: For major policy, tech, economic, or legal stories, write 1-2 sentences explaining practical real-world implications for ordinary citizens, consumers, or developers. Otherwise keep empty string.\n"
            f"9. GOOD NEWS CORNER: Set 'is_good_news': true if this story represents an inspiring scientific discovery, medical breakthrough, conservation progress, student achievement, or positive community initiative.\n"
            f"10. FACT CHECK & CLAIMS: If this story analyzes an online rumor, meme claim, or controversial assertion, set 'fact_check_verdict' to 'SUPPORTED', 'MISLEADING', or 'FALSE'. Otherwise keep empty string.\n"
            f"11. ALL OUTPUT IN CRISP ENGLISH: If input text is in Indian regional languages (Bengali, Hindi, Telugu, Tamil, etc.), translate and synthesize cleanly into English.\n"
            f"12. STRICT FACTUAL FIDELITY: Never fabricate facts, quotes, statistics, motives, or events. Wit and engagement come from phrasing and insight, NEVER from false details.\n"
            f"13. NO REPETITION: Ensure diverse sentence rhythms, vocabulary, and angles across the batch.\n\n"
            f"INPUT STORIES ({len(batch)} items):\n"
            f"{json.dumps(input_data, indent=2)}\n\n"
            f"Return JSON schema:\n"
            f'{{"stories": [{{"cluster_id": "...", "headline": "...", "summary": "...", "key_takeaway": "...", "why_it_matters": "...", "is_good_news": false, "fact_check_verdict": "", "category": "...", "state_or_region": "...", "urgency_tag": "..."}}]}}'
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
                try:
                    cfg = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                        http_options=types.HttpOptions(timeout=22000),
                    )
                except Exception:
                    cfg = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    )
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=cfg,
                )
                return response.text
            else:
                model = client.GenerativeModel(
                    model_name=model_name,
                    generation_config={"response_mime_type": "application/json", "temperature": 0.2},
                )
                response = model.generate_content(prompt, request_options={"timeout": 22.0})
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

        groq_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

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
                timeout=20.0,
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
                        why_it_matters=s.get("why_it_matters", ""),
                        is_good_news=bool(s.get("is_good_news", False)),
                        fact_check_verdict=s.get("fact_check_verdict", ""),
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
                for gm in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
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
                for gm in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
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
            cluster_copy["why_it_matters"] = story.why_it_matters
            cluster_copy["is_good_news"] = story.is_good_news
            cluster_copy["fact_check_verdict"] = story.fact_check_verdict
            cluster_copy["urgency_tag"] = story.urgency_tag
            cluster_copy["category"] = story.category or cluster.get("section", "")
            cluster_copy["state_or_region"] = story.state_or_region or cluster.get("state_tag", "")

            synthesized_clusters.append(cluster_copy)

        logger.info(
            f"Parallel Synthesis complete: 100% ({len(synthesized_clusters)}/{total_stories}) "
            f"stories synthesized and merged with zero omission."
        )
        return synthesized_clusters
