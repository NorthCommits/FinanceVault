"""
financevault/reranking/cross_encoder.py

Cross-encoder reranker for FinanceVault.

Takes the hybrid retriever's top-20 candidates and scores each
(query, chunk) pair using a cross-encoder model, returning the
final top-k (default 5) for generation.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - 6-layer MiniLM, fast on CPU (~1-2s for 20 candidates)
    - Trained on MS MARCO passage ranking (standard IR benchmark)
    - Scores normalised to [0, 1] via sigmoid (Option B)
    - Max input: 512 tokens — we truncate chunk text to 400 tokens

Why cross-encoder over bi-encoder for reranking:
    Bi-encoders (FAISS) encode query and chunk independently.
    Cross-encoders encode the (query, chunk) pair jointly — full
    attention across both — which is far more accurate but 100x slower.
    Running it on 20 candidates (not 1000) keeps latency acceptable.

Design:
    - Model loaded once as a module-level singleton
    - Sigmoid normalisation applied to all raw logit scores
    - Chunk text truncated to 400 tokens before cross-encoder input
    - RRF score used as tiebreaker when cross-encoder scores are equal
    - Never raises — on model failure returns input list re-sorted by RRF
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import tiktoken

from ..retrieval.hybrid_retriever import RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CROSS_ENCODER_MODEL  = "BAAI/bge-reranker-large"
MAX_CHUNK_TOKENS     = 200     # Truncate chunk text before cross-encoder input
DEFAULT_TOP_K        = 5       # Final candidates passed to generation


# ---------------------------------------------------------------------------
# Tokenizer for truncation
# ---------------------------------------------------------------------------
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _truncate(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> str:
    """Truncate text to max_tokens tokens using GPT-4o tokenizer."""
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _TOKENIZER.decode(tokens[:max_tokens])


# ---------------------------------------------------------------------------
# Sigmoid normalisation
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Apply sigmoid to map unbounded logit to [0, 1]."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# ---------------------------------------------------------------------------
# Model singleton
# Loaded once on first call to rerank(), reused for all subsequent calls
# ---------------------------------------------------------------------------
_cross_encoder = None


def _load_cross_encoder():
    """
    Load the cross-encoder model as a module-level singleton.
    Returns the model or None if loading fails.
    """
    global _cross_encoder

    if _cross_encoder is not None:
        return _cross_encoder

    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"[reranker] Loading CrossEncoder ({CROSS_ENCODER_MODEL})...")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info("[reranker] CrossEncoder loaded successfully.")
        return _cross_encoder

    except Exception as e:
        logger.error(
            f"[reranker] Failed to load CrossEncoder: {type(e).__name__}: {e}. "
            f"Reranker will fall back to RRF ordering."
        )
        return None


# ---------------------------------------------------------------------------
# Core reranker function
# ---------------------------------------------------------------------------

def rerank(
    query       : str,
    results     : List[RetrievalResult],
    top_k       : int = DEFAULT_TOP_K,
    model       : Optional[object] = None,
) -> List[RetrievalResult]:
    """
    Rerank a list of RetrievalResults using a cross-encoder.

    For each result, the cross-encoder scores the (query, chunk_text) pair
    jointly. Scores are sigmoid-normalised to [0, 1] and attached to each
    RetrievalResult as `cross_encoder_score`.

    Results are sorted by descending cross_encoder_score. The top_k are
    returned. RRF score is used as a tiebreaker.

    Args:
        query:   The user's natural language query string.
        results: List of RetrievalResult from HybridRetriever.retrieve().
                 Typically 20 candidates.
        top_k:   Number of results to return. Default 5 for generation.
        model:   Optional pre-loaded CrossEncoder instance. If None, the
                 module singleton is used (loaded on first call).

    Returns:
        List of RetrievalResult sorted by descending cross_encoder_score,
        length <= top_k. Each result has cross_encoder_score set.

        On failure (model unavailable), returns the input list sorted by
        RRF score, truncated to top_k — so generation always gets something.

    Example:
        from financevault.reranking import rerank

        reranked = rerank(
            query   = "What was Apple's gross margin in fiscal 2024?",
            results = hybrid_results,   # List[RetrievalResult], len=20
            top_k   = 5,
        )
        for r in reranked:
            print(f"CE={r.cross_encoder_score:.3f} | {r.chunk.chunk_id}")
            print(r.chunk.text[:200])
    """
    if not results:
        logger.warning("[reranker] Empty results list. Nothing to rerank.")
        return []

    if not query.strip():
        logger.warning("[reranker] Empty query. Returning top_k by RRF score.")
        return sorted(results, key=lambda r: r.rrf_score, reverse=True)[:top_k]

    # Load model
    ce_model = model or _load_cross_encoder()

    # ------------------------------------------------------------------
    # Fallback: if model unavailable, return by RRF score
    # ------------------------------------------------------------------
    if ce_model is None:
        logger.warning(
            "[reranker] CrossEncoder unavailable. "
            "Returning top_k results sorted by RRF score."
        )
        fallback = sorted(results, key=lambda r: r.rrf_score, reverse=True)[:top_k]
        for r in fallback:
            r.cross_encoder_score = None
        return fallback

    # ------------------------------------------------------------------
    # Build (query, chunk_text) pairs for batch scoring
    # Truncate chunk text to stay within cross-encoder's token limit
    # ------------------------------------------------------------------
    pairs = [
        (query.strip(), _truncate(r.chunk.text))
        for r in results
    ]

    # ------------------------------------------------------------------
    # Score all pairs in one batch call
    # ------------------------------------------------------------------
    try:
        logger.info(
            f"[reranker] Scoring {len(pairs)} (query, chunk) pairs..."
        )
        raw_scores = ce_model.predict(pairs)

        # Normalise with sigmoid → [0, 1]
        norm_scores = [_sigmoid(float(s)) for s in raw_scores]

        # Attach scores to results
        for result, score in zip(results, norm_scores):
            result.cross_encoder_score = round(score, 4)

        # Sort by descending cross_encoder_score, RRF as tiebreaker
        ranked = sorted(
            results,
            key=lambda r: (r.cross_encoder_score, r.rrf_score),
            reverse=True,
        )

        final = ranked[:top_k]

        # Log the reranking result
        _log_reranking(query, results, final)

        return final

    except Exception as e:
        logger.error(
            f"[reranker] Scoring failed: {type(e).__name__}: {e}. "
            f"Falling back to RRF ordering."
        )
        fallback = sorted(results, key=lambda r: r.rrf_score, reverse=True)[:top_k]
        for r in fallback:
            r.cross_encoder_score = None
        return fallback


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_reranking(
    query     : str,
    all_results : List[RetrievalResult],
    final       : List[RetrievalResult],
) -> None:
    """Log a compact before/after comparison showing reranking movement."""
    lines = [
        f"[reranker] RERANKING RESULT for: '{query[:70]}'",
        f"  {'Chunk ID':<40} {'RRF':>8}  {'CE Score':>9}  {'Movement'}",
        f"  {'-'*70}",
    ]

    # Build RRF rank lookup
    rrf_ranks = {
        r.chunk.chunk_id: i + 1
        for i, r in enumerate(
            sorted(all_results, key=lambda x: x.rrf_score, reverse=True)
        )
    }

    for ce_rank, r in enumerate(final, 1):
        rrf_rank = rrf_ranks.get(r.chunk.chunk_id, "?")
        movement = rrf_rank - ce_rank if isinstance(rrf_rank, int) else 0

        if movement > 0:
            arrow = f"↑ +{movement}"
        elif movement < 0:
            arrow = f"↓ {movement}"
        else:
            arrow = "  ="

        lines.append(
            f"  {r.chunk.chunk_id:<40} "
            f"{r.rrf_score:>8.4f}  "
            f"{r.cross_encoder_score:>9.4f}  "
            f"#{ce_rank} (was #{rrf_rank}) {arrow}"
        )

    logger.info("\n".join(lines))