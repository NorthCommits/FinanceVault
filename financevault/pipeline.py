"""
financevault/pipeline.py

Main orchestrator for FinanceVault.

Two modes:

    BUILD mode — call pipeline.build()
        Loads parsed sections from data/processed/all_sections.json
        Runs adaptive chunking (selector picks best strategy per section)
        Embeds all chunks using OpenAI text-embedding-3-small
        Builds FAISS + BM25 indexes
        Saves everything to data/indexes/

    QUERY mode — call pipeline.query()
        Loads indexes from data/indexes/ (if not already loaded)
        Runs hybrid retrieval (BM25 + FAISS → RRF fusion)
        Reranks top-20 with cross-encoder
        Generates cited answer with GPT-4o
        Returns VerifiedResponse

The pipeline object maintains state between queries so indexes
are loaded once and reused across all subsequent queries.

Usage:

    from financevault.pipeline import FinanceVaultPipeline
    from openai import OpenAI

    client   = OpenAI()
    pipeline = FinanceVaultPipeline(openai_client=client)

    # Build once
    pipeline.build()

    # Query many times
    response = pipeline.query("What was Apple's gross margin in FY2024?")
    print(response.display())

    # Query with filters
    response = pipeline.query(
        "What are the main risk factors?",
        filters={"ticker": "MSFT", "fiscal_year": 2024}
    )
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .generation.generator import generate
from .generation.verifier import VerifiedResponse, verify
from .ingestion.chunking.selector import select_chunks_batch
from .ingestion.chunking.strategies import StrategyConfig
from .ingestion.models import (
    Chunk,
    ChunkingStrategy,
    FilingMetadata,
    ParsedSection,
    SectionSignals,
    SectionType,
)
from .reranking.cross_encoder import rerank
from .retrieval.bm25_store import BM25Store
from .retrieval.embedder import embed_chunks
from .retrieval.faiss_store import FAISSStore
from .retrieval.hybrid_retriever import HybridRetriever

load_dotenv()

logger = logging.getLogger(__name__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR       = Path(__file__).parent.parent
DATA_DIR       = ROOT_DIR / "data"
PROCESSED_DIR  = DATA_DIR / "processed"
INDEX_DIR      = DATA_DIR / "indexes"
ALL_SECTIONS   = PROCESSED_DIR / "all_sections.json"
CHUNKS_FILE    = INDEX_DIR / "chunks.json"


# ---------------------------------------------------------------------------
# Section deserialisation
# Reconstruct ParsedSection objects from the JSON saved by fetch_data.py
# ---------------------------------------------------------------------------

def _load_sections_from_disk(path: Path = ALL_SECTIONS) -> List[ParsedSection]:
    """
    Load ParsedSection objects from the combined JSON file saved by fetch_data.py.
    This avoids re-fetching from EDGAR on every build.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Sections file not found at {path}. "
            f"Run fetch_data.py first to collect SEC filing data."
        )

    logger.info(f"[pipeline] Loading sections from {path}...")
    raw = json.loads(path.read_text(encoding="utf-8"))

    sections: List[ParsedSection] = []
    for d in raw:
        meta = FilingMetadata(
            company_name = d["metadata"]["company_name"],
            ticker       = d["metadata"]["ticker"],
            cik          = d["metadata"]["cik"],
            fiscal_year  = d["metadata"]["fiscal_year"],
            filing_date  = date.fromisoformat(d["metadata"]["filing_date"]),
            accession_no = d["metadata"]["accession_no"],
            sector       = d["metadata"].get("sector"),
            sic_code     = None,
        )
        signals = SectionSignals(
            token_count         = d["signals"]["token_count"],
            numerical_density   = d["signals"]["numerical_density"],
            table_density       = d["signals"]["table_density"],
            avg_sentence_length = d["signals"]["avg_sentence_length"],
            has_subsections     = d["signals"]["has_subsections"],
            table_row_count     = d["signals"]["table_row_count"],
            paragraph_count     = d["signals"]["paragraph_count"],
        )
        sections.append(
            ParsedSection(
                section_id   = d["section_id"],
                metadata     = meta,
                item_number  = d["item_number"],
                item_title   = d["item_title"],
                section_type = SectionType(d["section_type"]),
                text         = d["text"],
                tables       = d.get("tables", []),
                signals      = signals,
                source_url   = d.get("source_url"),
            )
        )

    logger.info(f"[pipeline] Loaded {len(sections)} sections from disk.")
    return sections


# ---------------------------------------------------------------------------
# Chunk serialisation
# Save and load chunks to avoid re-embedding on subsequent runs
# ---------------------------------------------------------------------------

def _save_chunks(chunks: List[Chunk], path: Path = CHUNKS_FILE) -> None:
    """Save all chunks (including embeddings) to disk as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for c in chunks:
        d = c.to_metadata_dict()
        d["text"]      = c.text
        d["embedding"] = c.embedding
        data.append(d)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[pipeline] {len(chunks)} chunks saved → {path}")


def _load_chunks(path: Path = CHUNKS_FILE) -> List[Chunk]:
    """Load chunks from disk, restoring embeddings."""
    if not path.exists():
        return []

    raw    = json.loads(path.read_text(encoding="utf-8"))
    chunks : List[Chunk] = []

    for d in raw:
        meta = FilingMetadata(
            company_name = d["company_name"],
            ticker       = d["ticker"],
            cik          = d["cik"],
            fiscal_year  = d["fiscal_year"],
            filing_date  = date.fromisoformat(d["filing_date"]),
            accession_no = d["accession_no"],
            sector       = d.get("sector"),
            sic_code     = None,
        )
        chunks.append(
            Chunk(
                chunk_id                = d["chunk_id"],
                section_id              = d["section_id"],
                metadata                = meta,
                item_number             = d["item_number"],
                item_title              = d["item_title"],
                section_type            = SectionType(d["section_type"]),
                text                    = d["text"],
                token_count             = d["token_count"],
                chunk_index             = d["chunk_index"],
                total_chunks_in_section = d["total_chunks_in_section"],
                chunking_strategy       = ChunkingStrategy(d["chunking_strategy"]),
                chunking_score          = d["chunking_score"],
                numerical_density       = d["numerical_density"],
                is_table_chunk          = d["is_table_chunk"],
                embedding               = d.get("embedding"),
                source_url              = d.get("source_url"),
            )
        )

    logger.info(f"[pipeline] Loaded {len(chunks)} chunks from {path}.")
    return chunks


# ---------------------------------------------------------------------------
# FinanceVaultPipeline
# ---------------------------------------------------------------------------

class FinanceVaultPipeline:
    """
    End-to-end FinanceVault pipeline.

    Manages two lifecycle phases:
        build()  — chunking → embedding → indexing (run once)
        query()  — retrieval → reranking → generation (run many times)

    The pipeline is stateful: after build() or the first query(),
    indexes are held in memory and reused across subsequent queries.
    """

    def __init__(
        self,
        openai_client   : Optional[OpenAI] = None,
        index_dir       : Path = INDEX_DIR,
        retriever_top_k : int = 20,
        reranker_top_k  : int = 5,
        use_llm_chunking: bool = True,
    ):
        """
        Args:
            openai_client:    Authenticated OpenAI client.
                              If None, reads OPENAI_API_KEY from .env.
            index_dir:        Where to save/load FAISS + BM25 indexes.
            retriever_top_k:  Candidates from hybrid retriever (default 20).
            reranker_top_k:   Final chunks passed to generation (default 5).
            use_llm_chunking: Whether to enable LLM_REGEX chunking strategy.
                              Adds cost (~$0.50-$1 per full build) but improves
                              chunking quality for complex sections.
        """
        self.client          = openai_client or OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.index_dir       = Path(index_dir)
        self.retriever_top_k = retriever_top_k
        self.reranker_top_k  = reranker_top_k
        self.use_llm_chunking= use_llm_chunking

        # Runtime state — populated by build() or _ensure_loaded()
        self._chunks    : List[Chunk]              = []
        self._faiss     : Optional[FAISSStore]     = None
        self._bm25      : Optional[BM25Store]      = None
        self._retriever : Optional[HybridRetriever]= None
        self._is_built  : bool                     = False

    # ------------------------------------------------------------------
    # BUILD
    # ------------------------------------------------------------------

    def build(
        self,
        sections_path  : Path = ALL_SECTIONS,
        force_rechunk  : bool = False,
        force_reembed  : bool = False,
    ) -> None:
        """
        Build the FinanceVault index from parsed sections.

        Steps:
            1. Load sections from data/processed/all_sections.json
            2. Run adaptive chunking (or load cached chunks if available)
            3. Embed chunks with OpenAI (or skip already-embedded chunks)
            4. Build FAISS + BM25 indexes
            5. Save everything to data/indexes/

        Args:
            sections_path: Path to all_sections.json from fetch_data.py.
            force_rechunk: If True, re-run chunking even if cached chunks exist.
            force_reembed: If True, re-embed all chunks even if embeddings exist.
        """
        logger.info("=" * 60)
        logger.info("[pipeline] BUILD START")
        logger.info("=" * 60)

        # ------------------------------------------------------------------
        # Step 1: Load or chunk
        # ------------------------------------------------------------------
        cached_chunks = _load_chunks() if not force_rechunk else []

        if cached_chunks and not force_rechunk:
            logger.info(
                f"[pipeline] Using {len(cached_chunks)} cached chunks. "
                f"Pass force_rechunk=True to re-run chunking."
            )
            self._chunks = cached_chunks
        else:
            logger.info("[pipeline] Step 1/4: Loading sections from disk...")
            sections = _load_sections_from_disk(sections_path)

            logger.info("[pipeline] Step 2/4: Running adaptive chunking...")
            config = StrategyConfig(
                openai_client = self.client if self.use_llm_chunking else None
            )
            self._chunks = select_chunks_batch(sections, config=config)
            logger.info(
                f"[pipeline] Chunking complete: {len(self._chunks)} chunks "
                f"from {len(sections)} sections."
            )

        # ------------------------------------------------------------------
        # Step 2: Embed
        # ------------------------------------------------------------------
        logger.info("[pipeline] Step 3/4: Embedding chunks with OpenAI...")
        self._chunks = embed_chunks(
            chunks        = self._chunks,
            client        = self.client,
            skip_existing = not force_reembed,
        )

        # Save chunks with embeddings
        _save_chunks(self._chunks)

        # ------------------------------------------------------------------
        # Step 3: Build indexes
        # ------------------------------------------------------------------
        logger.info("[pipeline] Step 4/4: Building FAISS + BM25 indexes...")

        self._faiss = FAISSStore(index_dir=self.index_dir)
        self._faiss.build(self._chunks)
        self._faiss.save()

        self._bm25 = BM25Store(index_dir=self.index_dir)
        self._bm25.build(self._chunks)
        self._bm25.save()

        # ------------------------------------------------------------------
        # Step 4: Wire retriever
        # ------------------------------------------------------------------
        self._retriever = HybridRetriever(
            faiss_store   = self._faiss,
            bm25_store    = self._bm25,
            openai_client = self.client,
            bm25_top_k    = self.retriever_top_k * 3,
            faiss_top_k   = self.retriever_top_k * 3,
        )
        self._is_built = True

        logger.info("=" * 60)
        logger.info("[pipeline] BUILD COMPLETE")
        logger.info(f"  Chunks indexed : {len(self._chunks)}")
        logger.info(f"  FAISS vectors  : {self._faiss.total_chunks}")
        logger.info(f"  BM25 documents : {self._bm25.total_chunks}")
        logger.info(f"  Index dir      : {self.index_dir.resolve()}")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------

    def query(
        self,
        question : str,
        filters  : Optional[Dict] = None,
        top_k    : Optional[int]  = None,
    ) -> VerifiedResponse:
        """
        Answer a financial question using the full RAG pipeline.

        Steps:
            1. Hybrid retrieval (BM25 + FAISS → RRF, top-20)
            2. Cross-encoder reranking (top-20 → top-5)
            3. GPT-4o generation with cited context
            4. Citation validation + confidence scoring
            5. Return VerifiedResponse

        Args:
            question: Natural language financial question.
            filters:  Optional metadata filters for retrieval.
                      Keys: ticker, fiscal_year, item_number,
                            section_type, is_table_chunk, sector
                      Example: {"ticker": "AAPL", "fiscal_year": 2024}
            top_k:    Override default reranker_top_k for this query.

        Returns:
            VerifiedResponse with answer, citations, confidence, audit trail.

        Example:
            response = pipeline.query(
                "What was Apple's revenue and net income in FY2024?",
                filters={"ticker": "AAPL"}
            )
            print(response.display())
        """
        if not question.strip():
            raise ValueError("[pipeline] Question cannot be empty.")

        # Load indexes if not already built
        self._ensure_loaded()

        reranker_k = top_k or self.reranker_top_k

        logger.info(f"\n[pipeline] QUERY: {question[:100]}")
        if filters:
            logger.info(f"[pipeline] Filters: {filters}")

        # ------------------------------------------------------------------
        # Step 1: Hybrid retrieval
        # ------------------------------------------------------------------
        retrieval_results = self._retriever.retrieve(
            query   = question,
            top_k   = self.retriever_top_k,
            filters = filters,
        )

        if not retrieval_results:
            logger.warning("[pipeline] Retrieval returned no results.")

        # ------------------------------------------------------------------
        # Step 2: Reranking
        # ------------------------------------------------------------------
        reranked = rerank(
            query   = question,
            results = retrieval_results,
            top_k   = reranker_k,
        )

        # ------------------------------------------------------------------
        # Step 3 + 4: Generation + Verification
        # ------------------------------------------------------------------
        gen_result = generate(
            query   = question,
            results = reranked,
            client  = self.client,
        )

        response = verify(gen_result)

        logger.info(
            f"[pipeline] Response ready. "
            f"Confidence: {response.confidence:.0%} | "
            f"Sources cited: {len(response.sources)} | "
            f"Tokens: {response.tokens_used}"
        )

        return response

    # ------------------------------------------------------------------
    # Ensure indexes loaded
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """
        Load indexes from disk if not already in memory.
        Called automatically before the first query().
        Allows using the pipeline in query-only mode without calling build().
        """
        if self._is_built:
            return

        logger.info("[pipeline] Indexes not in memory. Loading from disk...")

        # Load chunks first (needed for BM25 chunk_map)
        self._chunks = _load_chunks()
        if not self._chunks:
            raise RuntimeError(
                "[pipeline] No chunks found in data/indexes/chunks.json. "
                "Run pipeline.build() first."
            )

        # Load FAISS
        self._faiss = FAISSStore(index_dir=self.index_dir)
        self._faiss.load()

        # Load BM25
        self._bm25 = BM25Store(index_dir=self.index_dir)
        self._bm25.load(self._chunks)

        # Wire retriever
        self._retriever = HybridRetriever(
            faiss_store   = self._faiss,
            bm25_store    = self._bm25,
            openai_client = self.client,
            bm25_top_k    = self.retriever_top_k * 3,
            faiss_top_k   = self.retriever_top_k * 3,
        )

        self._is_built = True
        logger.info(
            f"[pipeline] Indexes loaded. "
            f"{len(self._chunks)} chunks | "
            f"{self._faiss.total_chunks} FAISS vectors | "
            f"{self._bm25.total_chunks} BM25 documents."
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def get_stats(self) -> Dict:
        """Return a summary of the current pipeline state."""
        return {
            "is_built"   : self._is_built,
            "chunks"     : len(self._chunks),
            "faiss_vecs" : self._faiss.total_chunks if self._faiss else 0,
            "bm25_docs"  : self._bm25.total_chunks  if self._bm25  else 0,
            "index_dir"  : str(self.index_dir.resolve()),
        }