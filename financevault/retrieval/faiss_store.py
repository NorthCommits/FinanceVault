"""
financevault/retrieval/faiss_store.py

Builds, saves, loads, and searches the FAISS dense vector index.

Index type: IndexFlatIP (exact inner product search)
    - Exact search, no approximation
    - Inner product on L2-normalised vectors == cosine similarity
    - Correct choice for ~1000 chunks; switch to IndexHNSWFlat at 100k+

Persistence:
    - FAISS index  → data/indexes/faiss.index
    - Chunk metadata → data/indexes/chunks_meta.json
      (all Chunk fields except the embedding vector, which lives in the index)

Design:
    The FAISS index stores vectors by integer position (0, 1, 2, ...).
    We maintain a parallel list of Chunk metadata in the same order.
    When search returns positions [3, 7, 12], we look up chunks_meta[3],
    chunks_meta[7], chunks_meta[12] to reconstruct the full Chunk objects.

Metadata filtering:
    Before searching, we optionally build a filtered sub-index on the fly
    from only the chunks matching the filter criteria (company, year, section).
    This is Option A from our design discussion — filter then search.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

from ..ingestion.models import Chunk, FilingMetadata, SectionType, ChunkingStrategy
from .embedder import EMBEDDING_DIMENSIONS, build_embedding_matrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_INDEX_DIR  = Path("data/indexes")
FAISS_INDEX_FILE   = "faiss.index"
CHUNKS_META_FILE   = "chunks_meta.json"


# ---------------------------------------------------------------------------
# Serialisation helpers
# Chunk → dict and dict → Chunk (excluding embedding, which lives in FAISS)
# ---------------------------------------------------------------------------

def _chunk_to_dict(chunk: Chunk) -> dict:
    """Serialise a Chunk to a JSON-safe dict (embedding excluded)."""
    return chunk.to_metadata_dict() | {"text": chunk.text}


def _dict_to_chunk(d: dict) -> Chunk:
    """Reconstruct a Chunk from a serialised dict. Embedding set to None."""
    from datetime import date
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
    return Chunk(
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
        embedding               = None,
        source_url              = d.get("source_url"),
    )


# ---------------------------------------------------------------------------
# FAISSStore class
# ---------------------------------------------------------------------------

class FAISSStore:
    """
    Manages the FAISS dense vector index for FinanceVault.

    Typical usage:
        # Build once
        store = FAISSStore()
        store.build(chunks)
        store.save()

        # Use at retrieval time
        store = FAISSStore()
        store.load()
        results = store.search(query_vec, top_k=20)

        # With metadata filter
        results = store.search(
            query_vec, top_k=20,
            filters={"ticker": "AAPL", "fiscal_year": 2024}
        )
    """

    def __init__(self, index_dir: Path = DEFAULT_INDEX_DIR):
        self.index_dir   = Path(index_dir)
        self.index       : Optional[faiss.Index] = None
        self.chunks      : List[Chunk]            = []
        self._is_built   = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: List[Chunk]) -> None:
        """
        Build the FAISS index from a list of embedded Chunk objects.

        Chunks without embeddings are skipped with a warning.
        The index and chunk list are stored in memory; call save() to persist.

        Args:
            chunks: List of Chunk objects with .embedding populated.
        """
        # Only index chunks that have embeddings
        valid_chunks = [c for c in chunks if c.embedding is not None]

        if not valid_chunks:
            raise ValueError(
                "[faiss_store] No chunks have embeddings. "
                "Run embedder.embed_chunks() before building the index."
            )

        if len(valid_chunks) < len(chunks):
            logger.warning(
                f"[faiss_store] {len(chunks) - len(valid_chunks)} chunks "
                f"have no embedding and will be excluded from the index."
            )

        # Build embedding matrix
        matrix = build_embedding_matrix(valid_chunks)
        if matrix is None:
            raise ValueError("[faiss_store] Failed to build embedding matrix.")

        # Build FAISS index
        # IndexFlatIP: exact inner product search
        # On L2-normalised vectors this equals cosine similarity
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
        self.index.add(matrix)
        self.chunks    = valid_chunks
        self._is_built = True

        logger.info(
            f"[faiss_store] Index built: {self.index.ntotal} vectors, "
            f"{EMBEDDING_DIMENSIONS} dimensions."
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, index_dir: Optional[Path] = None) -> None:
        """
        Persist the FAISS index and chunk metadata to disk.

        Args:
            index_dir: Directory to save to. Defaults to self.index_dir.
        """
        if not self._is_built:
            raise RuntimeError("[faiss_store] Index not built. Call build() first.")

        save_dir = Path(index_dir or self.index_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS binary
        index_path = save_dir / FAISS_INDEX_FILE
        faiss.write_index(self.index, str(index_path))
        logger.info(f"[faiss_store] FAISS index saved → {index_path}")

        # Save chunk metadata as JSON (no embeddings — they live in FAISS)
        meta_path = save_dir / CHUNKS_META_FILE
        meta_list = [_chunk_to_dict(c) for c in self.chunks]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, ensure_ascii=False, indent=2)
        logger.info(
            f"[faiss_store] Chunk metadata saved → {meta_path} "
            f"({len(meta_list)} chunks)"
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, index_dir: Optional[Path] = None) -> None:
        """
        Load a previously saved FAISS index and chunk metadata from disk.

        Args:
            index_dir: Directory to load from. Defaults to self.index_dir.

        Raises:
            FileNotFoundError if the index files do not exist.
        """
        load_dir   = Path(index_dir or self.index_dir)
        index_path = load_dir / FAISS_INDEX_FILE
        meta_path  = load_dir / CHUNKS_META_FILE

        if not index_path.exists():
            raise FileNotFoundError(
                f"[faiss_store] FAISS index not found at {index_path}. "
                f"Run the ingestion pipeline first to build the index."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"[faiss_store] Chunk metadata not found at {meta_path}."
            )

        self.index = faiss.read_index(str(index_path))
        logger.info(
            f"[faiss_store] FAISS index loaded: {self.index.ntotal} vectors."
        )

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_list = json.load(f)
        self.chunks    = [_dict_to_chunk(d) for d in meta_list]
        self._is_built = True
        logger.info(
            f"[faiss_store] Chunk metadata loaded: {len(self.chunks)} chunks."
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vec : np.ndarray,
        top_k     : int = 20,
        filters   : Optional[Dict] = None,
    ) -> List[tuple[Chunk, float]]:
        """
        Search the FAISS index for the top-k most similar chunks.

        If filters are provided, a temporary sub-index is built from only
        the matching chunks before searching. This is Option A (filter then
        search) from our design discussion — more precise than post-filtering.

        Args:
            query_vec: L2-normalised query embedding from embed_query().
                       Shape: (EMBEDDING_DIMENSIONS,)
            top_k:     Number of results to return.
            filters:   Optional dict of metadata filters. Supported keys:
                           ticker       (str)   — e.g. "AAPL"
                           fiscal_year  (int)   — e.g. 2024
                           item_number  (str)   — e.g. "7"
                           section_type (str)   — e.g. "narrative"
                           is_table_chunk (bool)
                       All provided filters are ANDed together.

        Returns:
            List of (Chunk, score) tuples, sorted by descending similarity score.
            Score is cosine similarity in [0.0, 1.0] (higher = more similar).

        Example:
            results = store.search(
                query_vec,
                top_k=20,
                filters={"ticker": "AAPL", "fiscal_year": 2024}
            )
            for chunk, score in results:
                print(f"{score:.3f} | {chunk.chunk_id}")
        """
        if not self._is_built:
            raise RuntimeError(
                "[faiss_store] Index not loaded. Call build() or load() first."
            )

        query_vec = np.array(query_vec, dtype=np.float32).reshape(1, -1)

        # ------------------------------------------------------------------
        # Option A: Filter → then search within the filtered subset
        # ------------------------------------------------------------------
        if filters:
            filtered_chunks, filtered_indices = self._apply_filters(filters)

            if not filtered_chunks:
                logger.warning(
                    f"[faiss_store] Filters {filters} matched 0 chunks. "
                    f"Returning empty results."
                )
                return []

            # Build a temporary sub-index from filtered embeddings
            # We need to re-extract the embeddings for filtered chunks
            # We do this by searching the full index and mapping back
            # For efficiency: search full index with larger k, then filter
            search_k   = min(top_k * 10, self.index.ntotal)
            scores_raw, indices_raw = self.index.search(query_vec, search_k)

            results: List[tuple[Chunk, float]] = []
            filtered_set = set(filtered_indices)

            for idx, score in zip(indices_raw[0], scores_raw[0]):
                if idx == -1:
                    continue
                if idx in filtered_set:
                    results.append((self.chunks[idx], float(score)))
                if len(results) >= top_k:
                    break

            logger.debug(
                f"[faiss_store] Filtered search: {len(filtered_chunks)} candidates → "
                f"{len(results)} results returned."
            )
            return results

        # ------------------------------------------------------------------
        # No filters: search full index
        # ------------------------------------------------------------------
        actual_k = min(top_k, self.index.ntotal)
        scores_raw, indices_raw = self.index.search(query_vec, actual_k)

        results = [
            (self.chunks[idx], float(score))
            for idx, score in zip(indices_raw[0], scores_raw[0])
            if idx != -1
        ]

        logger.debug(
            f"[faiss_store] Full index search: {len(results)} results returned."
        )
        return results

    # ------------------------------------------------------------------
    # Filter helper
    # ------------------------------------------------------------------

    def _apply_filters(
        self, filters: Dict
    ) -> tuple[List[Chunk], List[int]]:
        """
        Return (filtered_chunks, their_indices) for chunks matching all filters.
        All filter keys are ANDed.
        """
        matched_chunks  : List[Chunk] = []
        matched_indices : List[int]   = []

        for i, chunk in enumerate(self.chunks):
            if self._matches(chunk, filters):
                matched_chunks.append(chunk)
                matched_indices.append(i)

        return matched_chunks, matched_indices

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
        return len(self.chunks)

    @property
    def is_built(self) -> bool:
        return self._is_built

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Look up a chunk by its chunk_id. Returns None if not found."""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None