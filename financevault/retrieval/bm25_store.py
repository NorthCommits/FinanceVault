"""
financevault/retrieval/bm25_store.py

Builds, saves, loads, and searches the BM25 sparse retrieval index.

Library: rank_bm25 (BM25Okapi variant)
    - Pure Python, no external dependencies beyond rank_bm25
    - BM25Okapi is the standard variant used in IR literature
    - Handles term frequency saturation and document length normalisation

Tokenisation:
    We use a simple whitespace + punctuation tokeniser with lowercase folding.
    We do NOT use a stemmer or stopword list because financial terms like
    "revenues", "revenue", "net" are semantically distinct and should not
    be collapsed. "EBITDA" and "ebitda" should match, so we lowercase.

Persistence:
    BM25 index serialised with pickle to data/indexes/bm25.pkl.
    Chunk IDs saved separately to data/indexes/bm25_chunk_ids.json
    so we can reconstruct which BM25 result maps to which Chunk.

Financial tokenisation enhancements:
    - Dollar amounts preserved: "$1,200" → "$1200" (comma stripped)
    - Percentages preserved: "12.5%" → "12.5%"
    - Ticker symbols kept uppercase: "AAPL" stays "AAPL"
    - Item numbers preserved: "Item 7" → "item_7" as single token
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from ..ingestion.models import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_INDEX_DIR  = Path("data/indexes")
BM25_INDEX_FILE    = "bm25.pkl"
BM25_IDS_FILE      = "bm25_chunk_ids.json"

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

# Preserve financial patterns before lowercasing
_DOLLAR_CLEAN   = re.compile(r"\$[\d,]+")
_ITEM_HEADER    = re.compile(r"\b(item)\s+(\d+[a-zA-Z]?)\b", re.IGNORECASE)
_PUNCTUATION    = re.compile(r"[^\w\$%\.]")
_MULTI_SPACE    = re.compile(r"\s+")


def _tokenise(text: str) -> List[str]:
    """
    Tokenise text for BM25 indexing with financial-domain enhancements.

    Steps:
        1. Normalise Item headers: "Item 7" → "item_7"
        2. Strip commas from dollar amounts: "$1,200" → "$1200"
        3. Lowercase
        4. Remove punctuation (keep $, %, .)
        5. Split on whitespace
        6. Remove empty tokens and single characters (except $-amounts)
    """
    # Step 1: Normalise Item headers to single tokens
    text = _ITEM_HEADER.sub(lambda m: f"item_{m.group(2).lower()}", text)

    # Step 2: Strip commas from dollar amounts
    text = _DOLLAR_CLEAN.sub(lambda m: m.group(0).replace(",", ""), text)

    # Step 3: Lowercase
    text = text.lower()

    # Step 4: Replace punctuation with spaces (keep $, %, .)
    text = _PUNCTUATION.sub(" ", text)

    # Step 5: Split
    tokens = _MULTI_SPACE.sub(" ", text).strip().split()

    # Step 6: Filter
    tokens = [
        t for t in tokens
        if len(t) > 1 or t.startswith("$")
    ]

    return tokens


# ---------------------------------------------------------------------------
# BM25Store class
# ---------------------------------------------------------------------------

class BM25Store:
    """
    Manages the BM25 sparse retrieval index for FinanceVault.

    Typical usage:
        # Build once
        store = BM25Store()
        store.build(chunks)
        store.save()

        # Use at retrieval time
        store = BM25Store()
        store.load(all_chunks)   # Pass chunks for ID→Chunk lookup
        results = store.search("Apple revenue fiscal 2024", top_k=20)

        # With metadata filter
        results = store.search(
            "EBITDA margin",
            top_k=20,
            filters={"ticker": "AAPL"}
        )
    """

    def __init__(self, index_dir: Path = DEFAULT_INDEX_DIR):
        self.index_dir  = Path(index_dir)
        self.bm25       : Optional[BM25Okapi] = None
        self.chunk_ids  : List[str]           = []
        self._chunk_map : Dict[str, Chunk]    = {}
        self._is_built  = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: List[Chunk]) -> None:
        """
        Build the BM25 index from a list of Chunk objects.

        Each chunk's text is tokenised and added to the corpus.
        The index preserves the same ordering as the chunks list
        so BM25 result positions map directly to chunk_ids.

        Args:
            chunks: List of Chunk objects (embedding not required for BM25).
        """
        if not chunks:
            raise ValueError("[bm25_store] Cannot build index from empty chunk list.")

        logger.info(f"[bm25_store] Tokenising {len(chunks)} chunks...")

        corpus = [_tokenise(c.text) for c in chunks]

        # Build BM25Okapi index
        self.bm25      = BM25Okapi(corpus)
        self.chunk_ids = [c.chunk_id for c in chunks]

        # Build chunk_id → Chunk lookup for search results
        self._chunk_map = {c.chunk_id: c for c in chunks}
        self._is_built  = True

        logger.info(
            f"[bm25_store] BM25 index built: {len(chunks)} documents, "
            f"avg {sum(len(t) for t in corpus) / len(corpus):.0f} tokens/doc."
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, index_dir: Optional[Path] = None) -> None:
        """
        Persist the BM25 index and chunk ID list to disk.

        Args:
            index_dir: Directory to save to. Defaults to self.index_dir.
        """
        if not self._is_built:
            raise RuntimeError("[bm25_store] Index not built. Call build() first.")

        save_dir = Path(index_dir or self.index_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Pickle the BM25 object
        bm25_path = save_dir / BM25_INDEX_FILE
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)
        logger.info(f"[bm25_store] BM25 index saved → {bm25_path}")

        # Save chunk IDs as JSON
        ids_path = save_dir / BM25_IDS_FILE
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(self.chunk_ids, f)
        logger.info(f"[bm25_store] Chunk IDs saved → {ids_path}")

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        chunks    : List[Chunk],
        index_dir : Optional[Path] = None,
    ) -> None:
        """
        Load a previously saved BM25 index from disk.

        Args:
            chunks:    Full list of Chunk objects (needed to rebuild
                       the chunk_id → Chunk lookup map for search results).
            index_dir: Directory to load from. Defaults to self.index_dir.

        Raises:
            FileNotFoundError if the index files do not exist.
        """
        load_dir  = Path(index_dir or self.index_dir)
        bm25_path = load_dir / BM25_INDEX_FILE
        ids_path  = load_dir / BM25_IDS_FILE

        if not bm25_path.exists():
            raise FileNotFoundError(
                f"[bm25_store] BM25 index not found at {bm25_path}. "
                f"Run the ingestion pipeline first."
            )
        if not ids_path.exists():
            raise FileNotFoundError(
                f"[bm25_store] Chunk IDs not found at {ids_path}."
            )

        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
        logger.info(f"[bm25_store] BM25 index loaded from {bm25_path}.")

        with open(ids_path, "r", encoding="utf-8") as f:
            self.chunk_ids = json.load(f)
        logger.info(
            f"[bm25_store] Chunk IDs loaded: {len(self.chunk_ids)} entries."
        )

        # Rebuild chunk_id → Chunk lookup
        self._chunk_map = {c.chunk_id: c for c in chunks}
        self._is_built  = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query   : str,
        top_k   : int = 20,
        filters : Optional[Dict] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search the BM25 index for the top-k most relevant chunks.

        Args:
            query:   Natural language query string.
            top_k:   Number of results to return.
            filters: Optional metadata filters (same keys as FAISSStore).
                     Applied after BM25 scoring — we score all, then filter.
                     For BM25 the corpus is small enough that post-filtering
                     is fine (unlike FAISS where pre-filtering saves compute).

        Returns:
            List of (Chunk, bm25_score) tuples, sorted by descending score.
            Scores are raw BM25 values (not normalised to [0, 1]).
            The hybrid retriever uses rankings, not raw scores, so this is fine.

        Example:
            results = store.search("Apple revenue growth fiscal 2024", top_k=20)
            for chunk, score in results:
                print(f"{score:.2f} | {chunk.chunk_id}")
        """
        if not self._is_built:
            raise RuntimeError(
                "[bm25_store] Index not loaded. Call build() or load() first."
            )

        if not query.strip():
            logger.warning("[bm25_store] Empty query. Returning empty results.")
            return []

        query_tokens = _tokenise(query)

        if not query_tokens:
            logger.warning(
                "[bm25_store] Query tokenised to empty list. "
                "All tokens may have been filtered. Returning empty results."
            )
            return []

        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Build (chunk_id, score) pairs sorted by descending score
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results: List[Tuple[Chunk, float]] = []

        for idx, bm25_score in ranked:
            if bm25_score <= 0.0:
                break  # All remaining scores are zero — stop early

            chunk_id = self.chunk_ids[idx]
            chunk    = self._chunk_map.get(chunk_id)

            if chunk is None:
                logger.warning(
                    f"[bm25_store] chunk_id {chunk_id} in index but not in chunk_map. "
                    f"Index may be stale — rebuild."
                )
                continue

            if filters and not self._matches(chunk, filters):
                continue

            results.append((chunk, float(bm25_score)))

            if len(results) >= top_k:
                break

        logger.debug(
            f"[bm25_store] Query '{query[:60]}...' → {len(results)} results."
        )

        return results

    # ------------------------------------------------------------------
    # Filter helper (same interface as FAISSStore for consistency)
    # ------------------------------------------------------------------

    @staticmethod
    def _matches(chunk: Chunk, filters: Dict) -> bool:
        """Check whether a chunk satisfies all filter conditions."""
        for key, value in filters.items():
            if key == "ticker":
                if chunk.metadata.ticker.upper() != str(value).upper():
                    return False
            elif key == "fiscal_year":
                if chunk.metadata.fiscal_year != int(value):
                    return False
            elif key == "item_number":
                if chunk.item_number != str(value):
                    return False
            elif key == "section_type":
                if chunk.section_type.value != str(value):
                    return False
            elif key == "is_table_chunk":
                if chunk.is_table_chunk != bool(value):
                    return False
            elif key == "sector":
                if chunk.metadata.sector != str(value):
                    return False
        return True

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def total_chunks(self) -> int:
        return len(self.chunk_ids)

    @property
    def is_built(self) -> bool:
        return self._is_built