"""
financevault/retrieval/hybrid_retriever.py

Combines BM25 sparse and FAISS dense retrieval using Reciprocal Rank Fusion.

Reciprocal Rank Fusion (RRF):
    Proposed by Cormack, Clarke & Buettcher (SIGIR 2009).
    For each chunk appearing in either ranked list:

        RRF_score = 1 / (k + rank_in_bm25) + 1 / (k + rank_in_faiss)

    where k=60 is the smoothing constant from the original paper.
    Chunks not appearing in a list get rank = infinity → contribution = 0.

    Why RRF over score fusion?
        - No calibration needed: BM25 scores and cosine similarities
          are on completely different scales and cannot be summed directly.
        - Robust: a chunk ranked #1 in BM25 and #50 in FAISS beats one
          ranked #10 in both — the dual-signal agreement matters.
        - Proven: RRF consistently outperforms linear score combination
          in TREC and BEIR benchmarks.

Public API:
    HybridRetriever.retrieve(query, top_k, filters) -> List[RetrievalResult]

RetrievalResult carries the Chunk plus all retrieval signals
(bm25_rank, faiss_rank, rrf_score) for transparency and reranking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import re

from ..ingestion.models import Chunk
from .bm25_store import BM25Store
from .embedder import embed_query
from .faiss_store import FAISSStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query router — maps query intent to optimal section filters
# ---------------------------------------------------------------------------

_METRIC_TERMS = re.compile(
    r"\b(revenue|net income|earnings|profit|loss|margin|ebitda|assets|"
    r"liabilities|equity|cash flow|operating income|gross margin|"
    r"return on|roa|roe|eps|dividend|capital expenditure|capex|"
    r"net sales|total sales|operating expense)\b",
    re.IGNORECASE,
)

_RISK_TERMS = re.compile(
    r"\b(risk|liquidity|uncertainty|regulatory|compliance|litigation|"
    r"cyber|market risk|credit risk|interest rate|inflation|geopolit|"
    r"competition risk|operational risk|legal|lawsuit)\b",
    re.IGNORECASE,
)

_STRATEGY_TERMS = re.compile(
    r"\b(strategy|strategic|business model|competition|growth|expansion|"
    r"acquisition|investment|product|service|market position|outlook|"
    r"capital allocation|dividend policy|share repurchase)\b",
    re.IGNORECASE,
)

# Known company name → ticker mapping for parallel retrieval detection
_COMPANY_MAP = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
    "alphabet": "GOOGL", "jpmorgan": "JPM", "jp morgan": "JPM",
    "goldman": "GS", "goldman sachs": "GS", "blackrock": "BLK",
    "exxon": "XOM", "exxonmobil": "XOM", "chevron": "CVX",
    "walmart": "WMT", "amazon": "AMZN",
}


def _detect_companies(query: str) -> list[str]:
    """Detect company tickers mentioned in the query."""
    q_lower = query.lower()
    found = []
    # Check two-word names first
    for name, ticker in sorted(_COMPANY_MAP.items(), key=lambda x: -len(x[0])):
        if name in q_lower and ticker not in found:
            found.append(ticker)
    return found


def _route_query(query: str, filters: dict) -> list[dict]:
    """
    Analyse query intent and return a list of retrieval jobs.
    Each job is a dict with {filters, weight} where weight controls
    how much this job contributes to the final RRF pool.

    Returns a single job for simple queries, multiple jobs for
    comparative queries that span two companies.
    """
    detected_companies = _detect_companies(query)

    # Determine best item filter based on query terms
    if _METRIC_TERMS.search(query):
        item_hint = None  # Search all sections — metrics appear in both 7 and 8
    elif _RISK_TERMS.search(query):
        item_hint = "1A"
    elif _STRATEGY_TERMS.search(query):
        item_hint = None  # Item 1 and 7 both relevant
    else:
        item_hint = None

    # If filters already specify a ticker, use a single job
    if filters.get("ticker"):
        job_filters = dict(filters)
        if item_hint and not filters.get("item_number"):
            job_filters["item_number"] = item_hint
        return [{"filters": job_filters, "weight": 1.0}]

    # Comparative query: two companies detected → parallel jobs
    if len(detected_companies) >= 2:
        jobs = []
        for ticker in detected_companies[:2]:
            job_filters = dict(filters)
            job_filters["ticker"] = ticker
            if not filters.get("item_number"):
                # For metric queries with two companies, always search Item 8
                # since financial figures live there — add a second job for Item 7
                if _METRIC_TERMS.search(query):
                    # Job 1: Item 8 financial statements
                    item8_filters = dict(job_filters)
                    item8_filters["item_number"] = "8"
                    jobs.append({"filters": item8_filters, "weight": 1.0})
                    # Job 2: Item 7 MD&A (contains narrative with same figures)
                    item7_filters = dict(job_filters)
                    item7_filters["item_number"] = "7"
                    jobs.append({"filters": item7_filters, "weight": 0.8})
                else:
                    if item_hint:
                        job_filters["item_number"] = item_hint
                    jobs.append({"filters": job_filters, "weight": 1.0})
            else:
                jobs.append({"filters": job_filters, "weight": 1.0})
        return jobs

    # Single company detected
    if len(detected_companies) == 1:
        job_filters = dict(filters)
        job_filters["ticker"] = detected_companies[0]
        if item_hint and not filters.get("item_number"):
            job_filters["item_number"] = item_hint
        return [{"filters": job_filters, "weight": 1.0}]

    # No company detected — search across all with item hint
    job_filters = dict(filters)
    if item_hint and not filters.get("item_number"):
        job_filters["item_number"] = item_hint
    return [{"filters": job_filters, "weight": 1.0}]

# ---------------------------------------------------------------------------
# RRF smoothing constant — k=60 from the original paper
# Increasing k reduces the influence of top-ranked documents.
# ---------------------------------------------------------------------------
RRF_K = 60


# ---------------------------------------------------------------------------
# RetrievalResult — what the hybrid retriever returns
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    A single retrieval result carrying the Chunk and all retrieval signals.

    Fields beyond the Chunk are used by:
        - reranking/cross_encoder.py  (uses rrf_score as a prior)
        - generation/verifier.py      (uses bm25_rank/faiss_rank for citations)
        - evaluation/metrics.py       (uses all signals for analysis)
    """
    chunk       : Chunk
    rrf_score   : float          # Final RRF score (higher = more relevant)
    bm25_rank   : Optional[int]  # Rank in BM25 results (1-indexed, None if absent)
    faiss_rank  : Optional[int]  # Rank in FAISS results (1-indexed, None if absent)
    bm25_score  : Optional[float] = None  # Raw BM25 score
    faiss_score : Optional[float] = None  # Raw cosine similarity score

    def __repr__(self) -> str:
        return (
            f"RetrievalResult("
            f"chunk_id={self.chunk.chunk_id!r}, "
            f"rrf={self.rrf_score:.4f}, "
            f"bm25_rank={self.bm25_rank}, "
            f"faiss_rank={self.faiss_rank})"
        )


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Hybrid retriever combining BM25 + FAISS via Reciprocal Rank Fusion.

    Typical usage:
        from openai import OpenAI
        from financevault.retrieval import HybridRetriever, FAISSStore, BM25Store

        faiss_store = FAISSStore()
        faiss_store.load()

        bm25_store = BM25Store()
        bm25_store.load(all_chunks)

        retriever = HybridRetriever(
            faiss_store = faiss_store,
            bm25_store  = bm25_store,
            openai_client = OpenAI(),
        )

        results = retriever.retrieve(
            query   = "What was Apple's gross margin in fiscal 2024?",
            top_k   = 20,
            filters = {"ticker": "AAPL", "fiscal_year": 2024},
        )

        for r in results:
            print(f"RRF={r.rrf_score:.4f} | BM25={r.bm25_rank} | FAISS={r.faiss_rank}")
            print(r.chunk.text[:200])
    """

    def __init__(
        self,
        faiss_store   : FAISSStore,
        bm25_store    : BM25Store,
        openai_client,
        rrf_k         : int = RRF_K,
        bm25_top_k    : int = 50,   # Candidates to fetch from each system
        faiss_top_k   : int = 50,   # before fusing and cutting to final top_k
    ):
        self.faiss_store    = faiss_store
        self.bm25_store     = bm25_store
        self.openai_client  = openai_client
        self.rrf_k          = rrf_k
        self.bm25_top_k     = bm25_top_k
        self.faiss_top_k    = faiss_top_k

    # ------------------------------------------------------------------
    # Main retrieve method
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query   : str,
        top_k   : int = 20,
        filters : Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve the top-k most relevant chunks for a query using hybrid search.

        Args:
            query:   Natural language query string.
            top_k:   Number of results to return after fusion.
            filters: Optional metadata filters applied to both retrievers.
                     Supported keys: ticker, fiscal_year, item_number,
                     section_type, is_table_chunk, sector.

        Returns:
            List of RetrievalResult objects sorted by descending RRF score.
            Length <= top_k.
        """
        if not query.strip():
            logger.warning("[hybrid] Empty query. Returning empty results.")
            return []

        logger.info(
            f"[hybrid] Retrieving: '{query[:80]}...' | "
            f"top_k={top_k} | filters={filters}"
        )

        # ------------------------------------------------------------------
        # Step 1: Route query to optimal retrieval jobs
        # ------------------------------------------------------------------
        jobs = _route_query(query, filters or {})

        logger.info(
            f"[hybrid] Query routed to {len(jobs)} job(s): "
            f"{[j['filters'] for j in jobs]}"
        )

        # ------------------------------------------------------------------
        # Step 2: Run BM25 + FAISS for each job, collect all results
        # ------------------------------------------------------------------
        all_bm25  : list = []
        all_faiss : list = []

        for job in jobs:
            job_filters = job["filters"]
            bm25_res  = self._run_bm25(query, job_filters)
            faiss_res = self._run_faiss(query, job_filters)
            all_bm25.extend(bm25_res)
            all_faiss.extend(faiss_res)

        # Deduplicate by chunk_id keeping best score
        all_bm25  = self._dedup(all_bm25)
        all_faiss = self._dedup(all_faiss)

        # ------------------------------------------------------------------
        # Step 3: RRF fusion
        # ------------------------------------------------------------------
        fused = self._reciprocal_rank_fusion(all_bm25, all_faiss)

        # ------------------------------------------------------------------
        # Step 4: Return top_k
        # ------------------------------------------------------------------
        final = fused[:top_k]

        logger.info(
            f"[hybrid] Fusion complete: "
            f"BM25={len(all_bm25)} + FAISS={len(all_faiss)} → "
            f"fused={len(fused)} → returning top {len(final)}."
        )

        return final

    # ------------------------------------------------------------------
    # BM25 runner
    # ------------------------------------------------------------------



    @staticmethod
    def _dedup(
        results: list[tuple],
    ) -> list[tuple]:
        """Remove duplicate chunks keeping the one with the highest score."""
        seen: dict = {}
        for chunk, score in results:
            cid = chunk.chunk_id
            if cid not in seen or score > seen[cid][1]:
                seen[cid] = (chunk, score)
        return list(seen.values())

    def _run_bm25(
        self,
        query   : str,
        filters : Optional[Dict],
    ) -> List[tuple[Chunk, float]]:
        """Run BM25 search and return ranked (chunk, score) list."""
        try:
            results = self.bm25_store.search(
                query   = query,
                top_k   = self.bm25_top_k,
                filters = filters,
            )
            logger.debug(f"[hybrid] BM25 returned {len(results)} candidates.")
            return results
        except Exception as e:
            logger.error(f"[hybrid] BM25 search failed: {e}. Continuing with FAISS only.")
            return []

    # ------------------------------------------------------------------
    # FAISS runner
    # ------------------------------------------------------------------

    def _run_faiss(
        self,
        query   : str,
        filters : Optional[Dict],
    ) -> List[tuple[Chunk, float]]:
        """Embed query and run FAISS search, returning ranked (chunk, score) list."""
        try:
            query_vec = embed_query(query, self.openai_client)
            results   = self.faiss_store.search(
                query_vec = query_vec,
                top_k     = self.faiss_top_k,
                filters   = filters,
            )
            logger.debug(f"[hybrid] FAISS returned {len(results)} candidates.")
            return results
        except Exception as e:
            logger.error(f"[hybrid] FAISS search failed: {e}. Continuing with BM25 only.")
            return []

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        bm25_results  : List[tuple[Chunk, float]],
        faiss_results : List[tuple[Chunk, float]],
    ) -> List[RetrievalResult]:
        """
        Combine BM25 and FAISS results using Reciprocal Rank Fusion.

        For each unique chunk appearing in either list:
            rrf_score = 1/(k + bm25_rank) + 1/(k + faiss_rank)

        Chunks only in BM25: faiss contribution = 0
        Chunks only in FAISS: bm25 contribution = 0

        Returns a list of RetrievalResult sorted by descending rrf_score.
        """
        # Build rank lookups: chunk_id → (rank_1indexed, raw_score)
        bm25_rank_map  : Dict[str, tuple[int, float]] = {
            chunk.chunk_id: (rank + 1, score)
            for rank, (chunk, score) in enumerate(bm25_results)
        }
        faiss_rank_map : Dict[str, tuple[int, float]] = {
            chunk.chunk_id: (rank + 1, score)
            for rank, (chunk, score) in enumerate(faiss_results)
        }

        # Collect all unique chunk_ids across both lists
        all_chunk_ids = set(bm25_rank_map.keys()) | set(faiss_rank_map.keys())

        # Build a chunk_id → Chunk lookup from both result lists
        chunk_lookup: Dict[str, Chunk] = {}
        for chunk, _ in bm25_results:
            chunk_lookup[chunk.chunk_id] = chunk
        for chunk, _ in faiss_results:
            chunk_lookup[chunk.chunk_id] = chunk

        # Compute RRF scores
        rrf_results: List[RetrievalResult] = []

        for chunk_id in all_chunk_ids:
            chunk = chunk_lookup[chunk_id]

            bm25_rank_val,  bm25_score_val  = bm25_rank_map.get(chunk_id,  (None, None))
            faiss_rank_val, faiss_score_val = faiss_rank_map.get(chunk_id, (None, None))

            rrf_score = 0.0
            if bm25_rank_val is not None:
                rrf_score += 1.0 / (self.rrf_k + bm25_rank_val)
            if faiss_rank_val is not None:
                rrf_score += 1.0 / (self.rrf_k + faiss_rank_val)

            rrf_results.append(
                RetrievalResult(
                    chunk       = chunk,
                    rrf_score   = round(rrf_score, 6),
                    bm25_rank   = bm25_rank_val,
                    faiss_rank  = faiss_rank_val,
                    bm25_score  = bm25_score_val,
                    faiss_score = faiss_score_val,
                )
            )

        # Sort by descending RRF score
        rrf_results.sort(key=lambda r: r.rrf_score, reverse=True)

        logger.debug(
            f"[hybrid] RRF fusion: {len(all_chunk_ids)} unique chunks fused. "
            f"Top score: {rrf_results[0].rrf_score:.6f} "
            f"(BM25={rrf_results[0].bm25_rank}, FAISS={rrf_results[0].faiss_rank})"
            if rrf_results else "[hybrid] RRF produced no results."
        )

        return rrf_results

    # ------------------------------------------------------------------
    # Convenience: retrieve and return just chunks (no metadata)
    # Used by simple callers that do not need retrieval signals
    # ------------------------------------------------------------------

    def retrieve_chunks(
        self,
        query   : str,
        top_k   : int = 20,
        filters : Optional[Dict] = None,
    ) -> List[Chunk]:
        """Thin wrapper returning only Chunk objects, not RetrievalResult."""
        return [r.chunk for r in self.retrieve(query, top_k, filters)]