"""
financevault/ingestion/models.py

All Pydantic models for FinanceVault.
Every file in the pipeline imports from here. Nothing else defines data shapes.

Hierarchy:
  FilingMetadata
  RawFiling          (wraps edgartools filing object + metadata)
  SectionSignals     (detected signals per section)
  ParsedSection      (one 10-K section, ready for chunking)
  ChunkingStrategy   (enum of available strategies)
  MetricScores       (8-metric scorecard for one strategy run)
  ChunkingResult     (one strategy's output: chunks + scores)
  Chunk              (final output unit, goes into the vector store)
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SectionType(str, Enum):
    """Content character of a 10-K section."""
    NARRATIVE        = "narrative"         # Prose-heavy: Item 1, 1A, 7 narrative
    FINANCIAL_TABLE  = "financial_table"   # Table-heavy: Item 8 statements
    MIXED            = "mixed"             # Both prose and tables: Item 7 MD&A


class ChunkingStrategy(str, Enum):
    """Available chunking strategies. Selector picks one per section."""
    RECURSIVE_600    = "recursive_600"     # Recursive splitter, 600-token target
    RECURSIVE_1100   = "recursive_1100"    # Recursive splitter, 1100-token target
    TABLE_AWARE      = "table_aware"       # Preserves financial table row integrity
    LLM_REGEX        = "llm_regex"         # GPT-4o generates document-specific regex


# ---------------------------------------------------------------------------
# Filing metadata — company identity, attached to every downstream model
# ---------------------------------------------------------------------------

class FilingMetadata(BaseModel):
    """
    Identity card for one SEC filing.
    Attached to every ParsedSection and Chunk so retrieval can filter
    by company, year, or sector without touching the text.
    """
    company_name  : str            = Field(..., description="Full legal company name")
    ticker        : str            = Field(..., description="Exchange ticker symbol, e.g. AAPL")
    cik           : str            = Field(..., description="SEC CIK number, zero-padded to 10 digits")
    fiscal_year   : int            = Field(..., description="Fiscal year this filing covers, e.g. 2024")
    filing_date   : date           = Field(..., description="Date the filing was submitted to EDGAR")
    accession_no  : str            = Field(..., description="EDGAR accession number, e.g. 0000320193-24-000123")
    sector        : Optional[str]  = Field(None, description="Industry sector, e.g. Technology")
    sic_code      : Optional[str]  = Field(None, description="SEC SIC code for the company")

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("cik")
    @classmethod
    def cik_zero_padded(cls, v: str) -> str:
        # SEC CIKs are always 10 digits with leading zeros
        return v.strip().zfill(10)


# ---------------------------------------------------------------------------
# Raw filing — output of edgar_fetcher.py
# ---------------------------------------------------------------------------

class RawFiling(BaseModel):
    """
    Direct output of edgar_fetcher.py.
    Wraps the edgartools filing object alongside our metadata.
    The `filing_object` field holds the raw edgartools TenK object.
    We store it as Any because edgartools objects are not Pydantic-serialisable;
    they are only used transiently inside document_parser.py.
    """
    metadata       : FilingMetadata
    filing_object  : Any           = Field(..., description="Raw edgartools TenK object")
    raw_html_url   : Optional[str] = Field(None, description="Direct URL to the HTML filing on EDGAR")

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Section signals — detected by document_parser.py per section
# ---------------------------------------------------------------------------

class SectionSignals(BaseModel):
    """
    Quantitative signals extracted from a section before chunking.
    These drive the weighted scoring in selector.py:
      - Narrative sections   → weight ICC, DCC, RC higher
      - Table sections       → weight TBI, BI higher
      - Mixed sections       → blend
    """
    token_count         : int   = Field(..., description="Total token count of the section text")
    numerical_density   : float = Field(..., ge=0.0, le=1.0,
                                        description="Fraction of tokens that are numeric ($, %, bps, digits)")
    table_density       : float = Field(..., ge=0.0, le=1.0,
                                        description="Fraction of lines that belong to a table row")
    avg_sentence_length : float = Field(..., description="Mean tokens per sentence in the section")
    has_subsections     : bool  = Field(..., description="True if section contains numbered sub-items")
    table_row_count     : int   = Field(0,  description="Total number of table rows detected")
    paragraph_count     : int   = Field(0,  description="Number of paragraph blocks in the section")


# ---------------------------------------------------------------------------
# Parsed section — output of document_parser.py
# ---------------------------------------------------------------------------

class ParsedSection(BaseModel):
    """
    One logical section of a 10-K after document_parser.py has processed it.
    This is the input unit to the chunking pipeline.

    section_id follows the pattern: {ticker}_{fiscal_year}_{item_slug}
    e.g. AAPL_2024_item_7
    """
    section_id     : str                    = Field(..., description="Unique ID for this section")
    metadata       : FilingMetadata
    item_number    : str                    = Field(..., description="10-K item number, e.g. '1', '1A', '7', '8'")
    item_title     : str                    = Field(..., description="Human-readable title, e.g. 'Risk Factors'")
    section_type   : SectionType
    text           : str                    = Field(..., description="Full cleaned text of the section")
    tables         : List[str]              = Field(default_factory=list,
                                                    description="Tables serialised as markdown strings, "
                                                                "one entry per table")
    signals        : SectionSignals
    source_url     : Optional[str]          = Field(None, description="URL of the source filing page")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Section text cannot be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# Metric scores — output of scorer.py for one strategy run
# ---------------------------------------------------------------------------

class MetricScores(BaseModel):
    """
    8-metric scorecard produced by scorer.py for one (section, strategy) pair.

    Paper metrics (5):
      sc   — Size Compliance
      icc  — Intrachunk Cohesion
      dcc  — Contextual Coherence (Downstream Context Coherence)
      bi   — Block Integrity
      rc   — Missing Reference Error (Coreference Completeness)

    FinanceVault additions (3):
      nds  — Numerical Density Score  (numeric tokens kept in-context)
      tbi  — Table Boundary Integrity (no row split across chunk boundaries)
      sps  — Section Purity Score     (no bleed across Item boundaries)

    All scores are in [0, 1]. Higher is better.
    weighted_total is computed by selector.py based on section_type weights.
    """
    sc             : float = Field(..., ge=0.0, le=1.0, description="Size Compliance")
    icc            : float = Field(..., ge=0.0, le=1.0, description="Intrachunk Cohesion")
    dcc            : float = Field(..., ge=0.0, le=1.0, description="Contextual Coherence")
    bi             : float = Field(..., ge=0.0, le=1.0, description="Block Integrity")
    rc             : float = Field(..., ge=0.0, le=1.0, description="Missing Reference Error")
    nds            : float = Field(..., ge=0.0, le=1.0, description="Numerical Density Score")
    tbi            : float = Field(..., ge=0.0, le=1.0, description="Table Boundary Integrity")
    sps            : float = Field(..., ge=0.0, le=1.0, description="Section Purity Score")
    weighted_total : float = Field(0.0, ge=0.0, le=1.0, description="Final weighted score, set by selector.py")

    def to_dict(self) -> Dict[str, float]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Chunking result — one strategy's output, produced inside selector.py
# ---------------------------------------------------------------------------

class ChunkingResult(BaseModel):
    """
    The output of running one chunking strategy on one ParsedSection.
    selector.py produces one ChunkingResult per strategy, then picks the winner.
    """
    strategy       : ChunkingStrategy
    chunks         : List[str]         = Field(..., description="Raw text of each chunk produced")
    scores         : MetricScores
    chunk_count    : int               = Field(..., description="Number of chunks produced")
    avg_chunk_tokens: float            = Field(..., description="Mean token count across chunks")


# ---------------------------------------------------------------------------
# Chunk — the final output unit, goes into FAISS + BM25 index
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """
    The atomic unit of FinanceVault.
    Every chunk stored in FAISS and BM25 carries this full structure.

    chunk_id follows: {ticker}_{fiscal_year}_{item_slug}_{index:04d}
    e.g. AAPL_2024_item_7_0003

    This metadata enables retrieval-time filtering without touching embeddings:
      - Filter by company, year, sector before vector search
      - Filter by section_type to restrict to tables or narrative
      - Filter by numerical_density to find data-rich chunks
      - Use chunking_score as a quality signal during reranking
    """
    chunk_id               : str              = Field(..., description="Unique chunk identifier")
    section_id             : str              = Field(..., description="Parent section ID")
    metadata               : FilingMetadata
    item_number            : str              = Field(..., description="Source 10-K item number")
    item_title             : str              = Field(..., description="Source 10-K item title")
    section_type           : SectionType
    text                   : str              = Field(..., description="Chunk text content")
    token_count            : int              = Field(..., description="Token count of this chunk")
    chunk_index            : int              = Field(..., description="0-based index within its section")
    total_chunks_in_section: int              = Field(..., description="Total chunks in the parent section")
    chunking_strategy      : ChunkingStrategy = Field(..., description="Strategy that produced this chunk")
    chunking_score         : float            = Field(..., ge=0.0, le=1.0,
                                                      description="Weighted quality score of the winning strategy")
    numerical_density      : float            = Field(..., ge=0.0, le=1.0,
                                                      description="Numerical density of this specific chunk")
    is_table_chunk         : bool             = Field(False,
                                                      description="True if chunk content is primarily a table")
    embedding              : Optional[List[float]] = Field(None,
                                                           description="OpenAI embedding vector, set after embed step")
    source_url             : Optional[str]    = Field(None, description="URL of source filing page")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chunk text cannot be empty")
        return v.strip()

    def to_metadata_dict(self) -> Dict[str, Any]:
        """
        Returns a flat dict of all metadata fields (excluding text and embedding).
        Used when storing chunks in FAISS alongside the index.
        """
        return {
            "chunk_id"               : self.chunk_id,
            "section_id"             : self.section_id,
            "company_name"           : self.metadata.company_name,
            "ticker"                 : self.metadata.ticker,
            "cik"                    : self.metadata.cik,
            "fiscal_year"            : self.metadata.fiscal_year,
            "filing_date"            : str(self.metadata.filing_date),
            "accession_no"           : self.metadata.accession_no,
            "sector"                 : self.metadata.sector,
            "item_number"            : self.item_number,
            "item_title"             : self.item_title,
            "section_type"           : self.section_type.value,
            "token_count"            : self.token_count,
            "chunk_index"            : self.chunk_index,
            "total_chunks_in_section": self.total_chunks_in_section,
            "chunking_strategy"      : self.chunking_strategy.value,
            "chunking_score"         : self.chunking_score,
            "numerical_density"      : self.numerical_density,
            "is_table_chunk"         : self.is_table_chunk,
            "source_url"             : self.source_url,
        }