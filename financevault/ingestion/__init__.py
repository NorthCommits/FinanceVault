"""
financevault/ingestion/__init__.py

Public API for the entire ingestion package.

This is the single import surface for pipeline.py.
Everything the pipeline needs from ingestion comes from here.

Usage:
    from financevault.ingestion import run_ingestion
    from financevault.ingestion import FilingMetadata, Chunk, ParsedSection
"""

from .edgar_fetcher import fetch_filings, fetch_filing, DEFAULT_TICKERS
from .document_parser import parse_filings, parse_filing
from .chunking import select_chunks_batch, StrategyConfig

# Re-export models so pipeline.py never imports from models.py directly
from .models import (
    FilingMetadata,
    RawFiling,
    ParsedSection,
    SectionSignals,
    SectionType,
    ChunkingStrategy,
    MetricScores,
    ChunkingResult,
    Chunk,
)


def run_ingestion(
    tickers     = None,
    fiscal_year = None,
    config      = None,
):
    """
    Full ingestion pipeline: fetch → parse → chunk.

    This is the single function pipeline.py calls to go from
    a list of tickers to a list of Chunk objects ready for embedding.

    Args:
        tickers:     List of ticker symbols. Defaults to DEFAULT_TICKERS.
        fiscal_year: Fiscal year to fetch. None = latest available.
        config:      StrategyConfig. None = defaults (no LLM_REGEX).

    Returns:
        List[Chunk] — all chunks across all companies and sections.

    Example:
        from openai import OpenAI
        from financevault.ingestion import run_ingestion, StrategyConfig

        config = StrategyConfig(openai_client=OpenAI())
        chunks = run_ingestion(
            tickers=["AAPL", "MSFT", "JPM"],
            fiscal_year=2024,
            config=config,
        )
        print(f"Total chunks ready for embedding: {len(chunks)}")
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    if config is None:
        config = StrategyConfig()

    # Step 1: Fetch raw filings from EDGAR
    raw_filings = fetch_filings(tickers, fiscal_year=fiscal_year)

    if not raw_filings:
        raise RuntimeError(
            "No filings fetched. Check your ticker list and network connection."
        )

    # Step 2: Parse each filing into sections
    sections = parse_filings(raw_filings)

    if not sections:
        raise RuntimeError(
            "No sections parsed from fetched filings. Check edgar_fetcher and document_parser logs."
        )

    # Step 3: Adaptive chunking — select best strategy per section
    chunks = select_chunks_batch(sections, config=config)

    return chunks


__all__ = [
    # Pipeline entry point
    "run_ingestion",

    # Fetch layer
    "fetch_filings",
    "fetch_filing",
    "DEFAULT_TICKERS",

    # Parse layer
    "parse_filings",
    "parse_filing",

    # Chunk layer
    "select_chunks_batch",
    "StrategyConfig",

    # Models
    "FilingMetadata",
    "RawFiling",
    "ParsedSection",
    "SectionSignals",
    "SectionType",
    "ChunkingStrategy",
    "MetricScores",
    "ChunkingResult",
    "Chunk",
]