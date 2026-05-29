"""
financevault/ingestion/chunking/selector.py

Orchestrates the full adaptive chunking pipeline for a single ParsedSection.

Flow per section:
    1. Load embedder (once, shared across all strategy runs)
    2. Determine which strategies to run (pre-filter based on section signals)
    3. Run each candidate strategy → List[str] of chunks
    4. Score each strategy → MetricScores
    5. Apply section-type-specific weights → weighted_total per strategy
    6. Select the winning strategy (highest weighted_total)
    7. Convert winning chunks → List[Chunk] with full metadata attached

Public API:
    select_chunks(section, config, embedder) -> List[Chunk]
    select_chunks_batch(sections, config)    -> List[Chunk]

The batch function is the main entry point used by pipeline.py.
It loads the embedder once and reuses it across all sections.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import tiktoken

from ..models import (
    Chunk,
    ChunkingResult,
    ChunkingStrategy,
    MetricScores,
    ParsedSection,
    SectionType,
)
from .scorer import (
    FINANCIAL_TABLE_WEIGHTS,
    MIXED_WEIGHTS,
    NARRATIVE_WEIGHTS,
    apply_weights,
    score,
)
from .strategies import StrategyConfig, run_strategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


# ---------------------------------------------------------------------------
# Numerical density (reused from strategies, kept local to avoid circular import)
# ---------------------------------------------------------------------------
_NUMERICAL_PATTERN = re.compile(
    r"\$[\d,]+|\d+\.?\d*\%|[\d,]+\s?bps|\d[\d,]*\.\d+|\b\d{4}\b|\b\d[\d,]{2,}\b"
)


def _numerical_density(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if _NUMERICAL_PATTERN.search(t))
    return hits / len(tokens)


# ---------------------------------------------------------------------------
# Strategy pre-filtering
# Decides which strategies are worth running for a given section
# Pre-filtering avoids wasting LLM calls on short sections
# ---------------------------------------------------------------------------

def _candidate_strategies(section: ParsedSection, config: StrategyConfig) -> List[ChunkingStrategy]:
    """
    Return the list of strategies to run for this section.

    Rules:
        - RECURSIVE_600 and RECURSIVE_1100 always run (cheap, no API cost)
        - TABLE_AWARE always runs for FINANCIAL_TABLE or MIXED sections
          For pure NARRATIVE with table_density < 0.1, TABLE_AWARE is skipped
        - LLM_REGEX only runs if section has >= llm_min_section_tokens tokens
          (not worth the API cost for short sections)
    """
    candidates = [ChunkingStrategy.RECURSIVE_600, ChunkingStrategy.RECURSIVE_1100]

    # TABLE_AWARE: include for table-heavy or mixed sections
    if (
        section.section_type in (SectionType.FINANCIAL_TABLE, SectionType.MIXED)
        or section.signals.table_density >= 0.10
    ):
        candidates.append(ChunkingStrategy.TABLE_AWARE)

    # LLM_REGEX: include only if section is long enough
    if section.signals.token_count >= config.llm_min_section_tokens:
        candidates.append(ChunkingStrategy.LLM_REGEX)

    return candidates


# ---------------------------------------------------------------------------
# Weight selection by section type
# ---------------------------------------------------------------------------

def _weights_for_section(section: ParsedSection) -> dict:
    """Return the appropriate weight dict for this section's type."""
    if section.section_type == SectionType.FINANCIAL_TABLE:
        return FINANCIAL_TABLE_WEIGHTS
    if section.section_type == SectionType.MIXED:
        return MIXED_WEIGHTS
    return NARRATIVE_WEIGHTS


# ---------------------------------------------------------------------------
# Chunk construction
# Converts raw chunk strings from the winning strategy into Chunk objects
# ---------------------------------------------------------------------------

def _build_chunk_id(ticker: str, fiscal_year: int, item_number: str, index: int) -> str:
    """
    Construct a deterministic, unique chunk ID.
    Format: {TICKER}_{FISCAL_YEAR}_item_{ITEM}_chunk_{INDEX:04d}
    Example: AAPL_2024_item_7_chunk_0003
    """
    clean_item = item_number.replace(" ", "_").lower()
    return f"{ticker}_{fiscal_year}_item_{clean_item}_chunk_{index:04d}"


def _chunks_to_models(
    raw_chunks       : List[str],
    section          : ParsedSection,
    winning_strategy : ChunkingStrategy,
    winning_score    : float,
) -> List[Chunk]:
    """
    Convert raw chunk strings into fully populated Chunk Pydantic models.

    Each chunk gets:
        - A unique chunk_id
        - Full FilingMetadata from its parent section
        - Section provenance (item_number, item_title, section_type)
        - Per-chunk computed signals (token_count, numerical_density, is_table_chunk)
        - Strategy provenance (which strategy won, what score it achieved)
        - Positional context (chunk_index, total_chunks_in_section)
        - Embedding placeholder (None — set later by retrieval/embedder.py)
    """
    total   = len(raw_chunks)
    ticker  = section.metadata.ticker
    year    = section.metadata.fiscal_year
    chunks  : List[Chunk] = []

    for i, text in enumerate(raw_chunks):
        if not text.strip():
            continue

        token_count  = _count_tokens(text)
        num_density  = _numerical_density(text)

        # A chunk is classified as a table chunk if more than 40% of its
        # lines contain pipe characters — it is primarily tabular content
        lines        = [l for l in text.splitlines() if l.strip()]
        table_lines  = sum(1 for l in lines if "|" in l)
        is_table     = (table_lines / len(lines)) >= 0.4 if lines else False

        chunk_id = _build_chunk_id(ticker, year, section.item_number, i)

        chunks.append(
            Chunk(
                chunk_id                = chunk_id,
                section_id              = section.section_id,
                metadata                = section.metadata,
                item_number             = section.item_number,
                item_title              = section.item_title,
                section_type            = section.section_type,
                text                    = text.strip(),
                token_count             = token_count,
                chunk_index             = i,
                total_chunks_in_section = total,
                chunking_strategy       = winning_strategy,
                chunking_score          = winning_score,
                numerical_density       = round(num_density, 4),
                is_table_chunk          = is_table,
                embedding               = None,
                source_url              = section.source_url,
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Core selector: one section at a time
# ---------------------------------------------------------------------------

def select_chunks(
    section  : ParsedSection,
    config   : StrategyConfig,
    embedder : Optional[object] = None,
) -> List[Chunk]:
    """
    Run adaptive chunking for a single ParsedSection.

    Steps:
        1. Determine candidate strategies based on section signals
        2. Run each strategy → List[str] chunks
        3. Score each strategy → MetricScores
        4. Apply section-type weights → weighted_total
        5. Select winner (highest weighted_total)
        6. Log the competition result
        7. Convert winning chunks → List[Chunk]

    Args:
        section:  A ParsedSection from document_parser.py
        config:   StrategyConfig with all tuneable parameters
        embedder: A loaded SentenceTransformer instance (or None)
                  If None, ICC and DCC use safe defaults (0.75)

    Returns:
        List[Chunk] — the winning strategy's chunks, fully populated.
        Empty list only if the section itself is empty (should not happen
        given document_parser.py's validation).
    """
    ticker      = section.metadata.ticker
    section_id  = section.section_id
    weights     = _weights_for_section(section)
    candidates  = _candidate_strategies(section, config)

    logger.info(
        f"[selector] {section_id} | type={section.section_type.value} | "
        f"tokens={section.signals.token_count} | "
        f"candidates={[c.value for c in candidates]}"
    )

    results: List[ChunkingResult] = []

    for strategy in candidates:
        try:
            # Run the strategy
            raw_chunks = run_strategy(strategy, section, config)

            if not raw_chunks:
                logger.warning(f"[selector] {strategy.value} produced no chunks for {section_id}. Skipping.")
                continue

            # Score the chunks
            metric_scores = score(raw_chunks, section, embedder)

            # Apply weights to get the weighted total
            weighted_total = apply_weights(metric_scores, weights)
            metric_scores  = metric_scores.model_copy(update={"weighted_total": weighted_total})

            avg_tokens = sum(_count_tokens(c) for c in raw_chunks) / len(raw_chunks)

            results.append(
                ChunkingResult(
                    strategy         = strategy,
                    chunks           = raw_chunks,
                    scores           = metric_scores,
                    chunk_count      = len(raw_chunks),
                    avg_chunk_tokens = round(avg_tokens, 1),
                )
            )

            logger.debug(
                f"[selector] {section_id} | {strategy.value:15s} | "
                f"chunks={len(raw_chunks):3d} | "
                f"avg_tokens={avg_tokens:6.0f} | "
                f"weighted={weighted_total:.4f} | "
                f"SC={metric_scores.sc:.2f} ICC={metric_scores.icc:.2f} "
                f"DCC={metric_scores.dcc:.2f} BI={metric_scores.bi:.2f} "
                f"RC={metric_scores.rc:.2f} NDS={metric_scores.nds:.2f} "
                f"TBI={metric_scores.tbi:.2f} SPS={metric_scores.sps:.2f}"
            )

        except Exception as e:
            logger.error(
                f"[selector] {section_id} | {strategy.value} raised {type(e).__name__}: {e}. "
                f"Skipping this strategy."
            )

    # ------------------------------------------------------------------
    # If every strategy failed (should be extremely rare), fall back to
    # a direct recursive_600 call with no scoring
    # ------------------------------------------------------------------
    if not results:
        logger.error(
            f"[selector] {section_id}: ALL strategies failed. "
            f"Using raw recursive_600 as last resort."
        )
        from .strategies import recursive_600
        fallback_chunks = recursive_600(section, config) or [section.text.strip()]
        return _chunks_to_models(
            raw_chunks       = fallback_chunks,
            section          = section,
            winning_strategy = ChunkingStrategy.RECURSIVE_600,
            winning_score    = 0.0,
        )

    # ------------------------------------------------------------------
    # Select the winner: highest weighted_total
    # Tie-break: prefer fewer chunks (simpler is better when scores tie)
    # ------------------------------------------------------------------
    winner = max(
        results,
        key=lambda r: (r.scores.weighted_total, -r.chunk_count),
    )

    # Log the competition summary
    _log_competition(section_id, results, winner)

    return _chunks_to_models(
        raw_chunks       = winner.chunks,
        section          = section,
        winning_strategy = winner.strategy,
        winning_score    = winner.scores.weighted_total,
    )


def _log_competition(
    section_id : str,
    results    : List[ChunkingResult],
    winner     : ChunkingResult,
) -> None:
    """Log a compact competition summary showing all strategies and the winner."""
    lines = [f"[selector] COMPETITION RESULT: {section_id}"]
    lines.append(f"  {'Strategy':<20} {'Score':>7}  {'Chunks':>6}  {'AvgTok':>7}")
    lines.append(f"  {'-'*48}")

    for r in sorted(results, key=lambda x: x.scores.weighted_total, reverse=True):
        marker = " ← WINNER" if r.strategy == winner.strategy else ""
        lines.append(
            f"  {r.strategy.value:<20} {r.scores.weighted_total:>7.4f}  "
            f"{r.chunk_count:>6}  {r.avg_chunk_tokens:>7.0f}{marker}"
        )

    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Batch selector — main entry point for the pipeline
# ---------------------------------------------------------------------------

def select_chunks_batch(
    sections : List[ParsedSection],
    config   : Optional[StrategyConfig] = None,
) -> List[Chunk]:
    """
    Run adaptive chunking across a list of ParsedSections.

    Loads the SentenceTransformer embedder once and reuses it for all sections.
    This is critical for performance — loading the model takes ~1-2 seconds
    and we do not want to pay that cost per section.

    Args:
        sections: Output of document_parser.parse_filings()
        config:   StrategyConfig. If None, uses defaults with no OpenAI client.
                  To enable LLM_REGEX, pass a config with openai_client set.

    Returns:
        Flat list of all Chunk objects across all sections, in section order.
        Sections that fail entirely are skipped with an error logged.

    Example:
        from openai import OpenAI
        from financevault.ingestion.chunking.strategies import StrategyConfig
        from financevault.ingestion.chunking.selector import select_chunks_batch

        config = StrategyConfig(openai_client=OpenAI())
        chunks = select_chunks_batch(sections, config)
        print(f"Total chunks: {len(chunks)}")
    """
    if config is None:
        config = StrategyConfig()

    # Load embedder once for the entire batch
    embedder = _load_embedder()

    all_chunks: List[Chunk] = []
    total      = len(sections)

    for i, section in enumerate(sections):
        ticker     = section.metadata.ticker
        item_num   = section.item_number
        section_id = section.section_id

        logger.info(
            f"[selector] Chunking [{i+1}/{total}]: "
            f"{ticker} Item {item_num} ({section.section_type.value}) — "
            f"{section.signals.token_count} tokens"
        )

        try:
            chunks = select_chunks(section, config, embedder)
            all_chunks.extend(chunks)
            logger.info(
                f"[selector] {section_id}: {len(chunks)} chunks produced "
                f"(running total: {len(all_chunks)})"
            )

        except Exception as e:
            logger.error(
                f"[selector] {section_id}: select_chunks raised unexpectedly: "
                f"{type(e).__name__}: {e}. Skipping section."
            )

    logger.info(
        f"[selector] Batch complete. "
        f"{len(all_chunks)} total chunks from {total} sections."
    )

    return all_chunks


# ---------------------------------------------------------------------------
# Embedder loader — called once per batch run
# ---------------------------------------------------------------------------

def _load_embedder():
    """
    Load the SentenceTransformer model used by the scorer for ICC and DCC.

    Model: all-MiniLM-L6-v2
        - 384 dimensions
        - Fast inference (~500ms per section on CPU)
        - Excellent cosine similarity quality for semantic coherence scoring
        - No API cost — runs locally

    Returns None if sentence-transformers is not installed or model fails to load.
    The scorer gracefully handles None embedder with safe defaults.
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("[selector] Loading SentenceTransformer (all-MiniLM-L6-v2)...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("[selector] Embedder loaded successfully.")
        return embedder
    except Exception as e:
        logger.warning(
            f"[selector] Could not load SentenceTransformer: {e}. "
            f"ICC and DCC will use safe defaults (0.75)."
        )
        return None