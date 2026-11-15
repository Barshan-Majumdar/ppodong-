"""
Autonomous 24/7 E-Paper News Agent - ML Deduplication & Editorial Priority Ranker
Clusters duplicate coverage across outlets using Sentence Transformers (all-MiniLM-L6-v2)
and cosine similarity (Agglomerative Clustering threshold >= 0.80).
Guarantees ZERO NEWS EXCLUSION and calculates deterministic Editorial Priority Scores.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("epaper.ml.deduplicator")

# Try importing ML dependencies; provide robust fallback if torch/sentence_transformers is absent
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class DeduplicatorRanker:
    """Performs semantic clustering and deterministic editorial ranking."""

    def __init__(self, config: Dict[str, Any], model_name: str = "all-MiniLM-L6-v2"):
        self.config = config
        self.pipeline_cfg = config.get("pipeline", {})
        self.weights_cfg = config.get("weights", {})
        self.sim_threshold = float(self.pipeline_cfg.get("deduplication_threshold", 0.80))
        self.distance_threshold = 1.0 - self.sim_threshold

        self.w_coverage = float(self.weights_cfg.get("coverage_weight", 3.0))
        self.regional_boosts = self.weights_cfg.get("regional_boost", {
            "kolkata_wb": 2.5,
            "indian_states": 1.5,
            "national": 1.5,
            "global": 1.0,
            "trends": 1.0,
        })
        self.authority_weights = self.weights_cfg.get("source_authority", {
            "high": 1.5,
            "standard": 1.0,
        })
        self.recency_max_bonus = float(self.weights_cfg.get("recency_max_bonus", 2.0))
        self.high_auth_domains = set(config.get("authority_domains", {}).get("high", []))

        # Lazy model loading
        self._model = None
        self._model_name = model_name

    def _get_embedding_model(self):
        if self._model is None and HAS_SENTENCE_TRANSFORMERS:
            try:
                logger.info(f"Loading SentenceTransformer model '{self._model_name}'...")
                self._model = SentenceTransformer(self._model_name)
                logger.info("SentenceTransformer loaded successfully.")
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer: {e}. Using TF-IDF fallback.")
                self._model = None
        return self._model

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute vector embeddings using SentenceTransformers or TF-IDF fallback."""
        model = self._get_embedding_model()
        if model is not None:
            try:
                embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
                return np.asarray(embeddings, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Embedding computation failed: {e}. Falling back to TF-IDF.")

        # Fallback to TF-IDF cosine space
        if HAS_SKLEARN:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            tfidf_mat = vectorizer.fit_transform(texts)
            return tfidf_mat.toarray()

        # Pure python dummy representation if no ML libs installed (for tests or ultra-minimal setups)
        return np.eye(len(texts), dtype=np.float32)

    def cluster_articles(self, articles: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group articles covering the exact same event into clusters.
        Zero news exclusion: Singletons remain clusters of length 1.
        """
        if not articles:
            return []
        if len(articles) == 1:
            return [[articles[0]]]

        # Build feature texts for embedding
        feature_texts = [
            f"{a.get('title', '')}. {a.get('lead', '')[:250]}" for a in articles
        ]

        embeddings = self.compute_embeddings(feature_texts)

        # Compute cluster labels
        n_samples = len(articles)
        if HAS_SKLEARN and n_samples > 1:
            try:
                clustering = AgglomerativeClustering(
                    metric="cosine",
                    linkage="average",
                    distance_threshold=self.distance_threshold,
                    n_clusters=None,
                )
                labels = clustering.fit_predict(embeddings)
            except Exception as e:
                logger.warning(f"AgglomerativeClustering failed: {e}. Falling back to pairwise cosine threshold.")
                # Manual single-linkage connected components on cosine similarity >= sim_threshold
                sim_matrix = cosine_similarity(embeddings)
                labels = [-1] * n_samples
                current_label = 0
                for i in range(n_samples):
                    if labels[i] != -1:
                        continue
                    labels[i] = current_label
                    for j in range(i + 1, n_samples):
                        if sim_matrix[i, j] >= self.sim_threshold:
                            labels[j] = current_label
                    current_label += 1
        else:
            # Fallback: each article is its own singleton
            labels = list(range(n_samples))

        # Aggregate articles into clusters
        cluster_map: Dict[int, List[Dict[str, Any]]] = {}
        for idx, label in enumerate(labels):
            if label not in cluster_map:
                cluster_map[label] = []
            cluster_map[label].append(articles[idx])

        clusters = list(cluster_map.values())
        logger.info(
            f"Clustering finished: {len(articles)} raw articles compressed into "
            f"{len(clusters)} distinct clusters (100% retention)."
        )
        return clusters

    def calculate_editorial_score(
        self,
        cluster: List[Dict[str, Any]],
        now_ts: float,
    ) -> float:
        """
        Deterministic Editorial Priority Ranking:
        Score(C) = (W_coverage * ln(1 + |C|)) + W_region + W_authority + W_recency
        """
        cluster_size = len(cluster)
        coverage_score = self.w_coverage * math.log(1 + cluster_size)

        # Determine dominant section and region
        regions = [a.get("region", "national") for a in cluster]
        sections = [a.get("section", "national") for a in cluster]

        # Prioritize kolkata_wb if present in cluster
        is_kolkata_wb = any(
            r == "kolkata_wb" or "kolkata" in (a.get("title", "") + a.get("lead", "")).lower()
            for a, r in zip(cluster, regions)
        )

        if is_kolkata_wb:
            region_boost = float(self.regional_boosts.get("kolkata_wb", 2.5))
        elif any(r == "indian_states" for r in regions):
            region_boost = float(self.regional_boosts.get("indian_states", 1.5))
        elif any(r == "national" for r in regions):
            region_boost = float(self.regional_boosts.get("national", 1.5))
        elif any(r == "global" for r in regions):
            region_boost = float(self.regional_boosts.get("global", 1.0))
        else:
            region_boost = float(self.regional_boosts.get("trends", 1.0))

        # Source authority check
        has_high_auth = any(
            a.get("is_high_authority", False) or a.get("domain", "") in self.high_auth_domains
            for a in cluster
        )
        auth_score = float(
            self.authority_weights.get("high" if has_high_auth else "standard", 1.0)
        )

        # Recency decay (newest article in cluster)
        timestamps = [a.get("timestamp", now_ts) for a in cluster]
        newest_ts = max(timestamps) if timestamps else now_ts
        age_hours = max(0.0, (now_ts - newest_ts) / 3600.0)

        # Linear decay from +2.0 at 0-2h to +0.0 at >= 24h
        if age_hours <= 2.0:
            recency_bonus = self.recency_max_bonus
        elif age_hours >= 26.0:
            recency_bonus = 0.0
        else:
            recency_bonus = self.recency_max_bonus * (1.0 - (age_hours - 2.0) / 24.0)

        total_score = coverage_score + region_boost + auth_score + recency_bonus
        return round(total_score, 3)

    def consolidate_cluster(
        self,
        cluster: List[Dict[str, Any]],
        now_ts: float,
        cluster_index: int,
    ) -> Dict[str, Any]:
        """Convert a list of raw articles into a consolidated story cluster."""
        # Pick primary article (prefer high authority and longest descriptive body)
        sorted_articles = sorted(
            cluster,
            key=lambda a: (
                1 if a.get("is_high_authority") else 0,
                len(a.get("text", "")),
            ),
            reverse=True,
        )
        primary = sorted_articles[0]

        # Consolidate sources and URLs
        sources = []
        domains = []
        urls = []
        canonical_urls = []

        for a in cluster:
            src = a.get("source")
            if src and src not in sources:
                sources.append(src)
            dom = a.get("domain")
            if dom and dom not in domains:
                domains.append(dom)
            u = a.get("url")
            if u and u not in urls:
                urls.append(u)
            cu = a.get("canonical_url") or u
            if cu and cu not in canonical_urls:
                canonical_urls.append(cu)

        score = self.calculate_editorial_score(cluster, now_ts)

        # Region tagging
        region = primary.get("region", "national")
        section = primary.get("section", "national")

        # Detect regional state if in indian_states section
        state_tag = ""
        full_text = f"{primary.get('title', '')} {primary.get('lead', '')}"
        states = [
            "Delhi", "Maharashtra", "Uttar Pradesh", "Bihar", "Odisha", "Assam", 
            "Punjab", "Haryana", "Rajasthan", "Gujarat", "Karnataka", "Tamil Nadu", 
            "Telangana", "Andhra Pradesh", "Kerala", "Madhya Pradesh", "Chhattisgarh", 
            "Goa", "Uttarakhand", "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", 
            "Manipur", "Meghalaya", "Tripura", "Sikkim", "Nagaland", "Arunachal Pradesh", "Mizoram"
        ]
        for st in states:
            if re.search(rf"\b{st}\b", full_text, re.IGNORECASE):
                state_tag = st.upper()
                break

        return {
            "cluster_id": f"CLUST_{cluster_index:04d}_{primary.get('id', '')[:8]}",
            "headline": primary.get("title", ""),
            "lead_summary": primary.get("lead", ""),
            "articles_count": len(cluster),
            "sources": sources,
            "domains": domains,
            "urls": urls,
            "canonical_urls": canonical_urls,
            "section": section,
            "region": region,
            "state_tag": state_tag,
            "primary_text": primary.get("text", "")[:1800],
            "score": score,
            "published_at": primary.get("published_at", datetime.now(timezone.utc).isoformat()),
            "timestamp": primary.get("timestamp", now_ts),
            "is_trend": primary.get("is_trend", False),
            "raw_articles": cluster,
        }

    def deduplicate_and_rank(
        self,
        articles: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Complete pipeline: Ingests raw articles -> clusters duplicate reporting ->
        computes priority scores -> returns ranked consolidated clusters (ZERO exclusion).
        """
        if not articles:
            return []

        now_ts = datetime.now(timezone.utc).timestamp()
        raw_clusters = self.cluster_articles(articles)

        ranked_clusters: List[Dict[str, Any]] = []
        for idx, cluster in enumerate(raw_clusters):
            consolidated = self.consolidate_cluster(cluster, now_ts, idx)
            ranked_clusters.append(consolidated)

        # Sort all clusters deterministically by score descending
        ranked_clusters.sort(key=lambda c: c["score"], reverse=True)

        logger.info(
            f"Deduplication & Ranking complete: {len(ranked_clusters)} ranked clusters ready. "
            f"Top score: {ranked_clusters[0]['score'] if ranked_clusters else 0.0}"
        )
        return ranked_clusters
