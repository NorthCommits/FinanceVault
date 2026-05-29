"""
financevault/retrieval/embedder.py

Handles all OpenAI embedding operations for FinanceVault.

Responsibilities:
    - Embed a list of Chunk objects (batch, with rate limit handling)
    - Embed a query string at retrieval time
    - Normalise all vectors to unit length (required for FAISS IndexFlatIP)
    - Cache embeddings to avoid re-calling the API on reruns

Model: text-embedding-3-small
    - 1536 dimensions
    - Significantly cheaper and faster than text-embedding-3-large
    - More than sufficient quality for financial RAG at our scale (~1000 chunks)

Design decisions:
    - Batching: OpenAI allows up to 2048 inputs per call, we use 512 for safety
    - Rate limit handling: exponential backoff on 429 errors
    - Normalisation: all vectors L2-normalised before storage so FAISS
      inner product == cosine similarity
    - Caching: embeddings saved alongside chunks so we never re-embed
      on subsequent pipeline runs unless forced
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import numpy as np
from openai import OpenAI, RateLimitError, APIError

from ..ingestion.models import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL      = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
BATCH_SIZE           = 100   # Chunks per OpenAI API call
MAX_RETRIES          = 5
INITIAL_BACKOFF      = 1.0   # seconds


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def _normalise(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a vector. Returns zero vector unchanged."""
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


# ---------------------------------------------------------------------------
# Single embedding call with retry
# ---------------------------------------------------------------------------

def _embed_batch(
    texts  : List[str],
    client : OpenAI,
    model  : str = EMBEDDING_MODEL,
) -> List[np.ndarray]:
    """
    Embed a batch of texts using OpenAI API with exponential backoff.

    Args:
        texts:  List of strings to embed (max 512 per call)
        client: Authenticated OpenAI client
        model:  Embedding model name

    Returns:
        List of L2-normalised numpy arrays, one per input text.

    Raises:
        RuntimeError if all retries are exhausted.
    """
    backoff = INITIAL_BACKOFF

    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                input = texts,
                model = model,
            )
            vectors = [
                _normalise(np.array(item.embedding, dtype=np.float32))
                for item in response.data
            ]
            return vectors

        except RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            logger.warning(
                f"[embedder] Rate limit hit. Retrying in {backoff:.1f}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})..."
            )
            time.sleep(backoff)
            backoff *= 2.0

        except APIError as e:
            if attempt == MAX_RETRIES - 1:
                raise
            logger.warning(
                f"[embedder] API error: {e}. Retrying in {backoff:.1f}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})..."
            )
            time.sleep(backoff)
            backoff *= 2.0

    raise RuntimeError(f"[embedder] All {MAX_RETRIES} embedding attempts failed.")


# ---------------------------------------------------------------------------
# Chunk embedder — main function used by pipeline.py
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks      : List[Chunk],
    client      : OpenAI,
    model       : str = EMBEDDING_MODEL,
    skip_existing: bool = True,
) -> List[Chunk]:
    """
    Embed all chunks and attach the embedding vector to each Chunk object.

    Processes in batches of BATCH_SIZE to stay within OpenAI limits.
    Updates each Chunk in-place by setting its `embedding` field.

    Args:
        chunks:        List of Chunk objects from the ingestion pipeline.
        client:        Authenticated OpenAI client.
        model:         Embedding model name.
        skip_existing: If True, skip chunks that already have embeddings.
                       Allows resuming interrupted runs without re-embedding.

    Returns:
        The same list of Chunk objects with `embedding` fields populated.
        Chunks that already had embeddings are returned unchanged if skip_existing=True.

    Example:
        from openai import OpenAI
        from financevault.retrieval.embedder import embed_chunks

        client = OpenAI()
        chunks = embed_chunks(chunks, client)
        print(chunks[0].embedding[:5])  # First 5 dimensions
    """
    # Separate chunks that need embedding from those already done
    to_embed   = [c for c in chunks if not skip_existing or c.embedding is None]
    already    = len(chunks) - len(to_embed)

    if already > 0:
        logger.info(f"[embedder] Skipping {already} already-embedded chunks.")

    if not to_embed:
        logger.info("[embedder] All chunks already embedded. Nothing to do.")
        return chunks

    logger.info(
        f"[embedder] Embedding {len(to_embed)} chunks "
        f"using {model} in batches of {BATCH_SIZE}..."
    )

    total_batches = (len(to_embed) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(to_embed), BATCH_SIZE):
        batch_chunks = to_embed[batch_idx : batch_idx + BATCH_SIZE]
        batch_texts  = [c.text for c in batch_chunks]
        batch_num    = (batch_idx // BATCH_SIZE) + 1

        logger.info(
            f"[embedder] Batch {batch_num}/{total_batches} "
            f"({len(batch_texts)} chunks)..."
        )

        try:
            vectors = _embed_batch(batch_texts, client, model)

            for chunk, vec in zip(batch_chunks, vectors):
                chunk.embedding = vec.tolist()

        except Exception as e:
            logger.error(
                f"[embedder] Batch {batch_num} failed permanently: {e}. "
                f"Affected chunks will have no embedding."
            )

        # Polite sleep between batches to avoid burst rate limits
        if batch_idx + BATCH_SIZE < len(to_embed):
            time.sleep(0.1)

    embedded_count = sum(1 for c in chunks if c.embedding is not None)
    logger.info(
        f"[embedder] Done. {embedded_count}/{len(chunks)} chunks now have embeddings."
    )

    return chunks


# ---------------------------------------------------------------------------
# Query embedder — used at retrieval time for every user query
# ---------------------------------------------------------------------------

def embed_query(
    query  : str,
    client : OpenAI,
    model  : str = EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Embed a single query string for retrieval.

    Returns a L2-normalised numpy array of shape (EMBEDDING_DIMENSIONS,).
    This vector is passed directly to faiss_store.search().

    Args:
        query:  The user's natural language query.
        client: Authenticated OpenAI client.
        model:  Embedding model name (must match the model used for chunks).

    Returns:
        np.ndarray of shape (1536,) — normalised query embedding.

    Example:
        vec = embed_query("What was Apple's revenue in fiscal 2024?", client)
        results = faiss_store.search(vec, top_k=20)
    """
    if not query.strip():
        raise ValueError("[embedder] Query string cannot be empty.")

    logger.debug(f"[embedder] Embedding query: {query[:80]}...")

    vectors = _embed_batch([query.strip()], client, model)
    return vectors[0]


# ---------------------------------------------------------------------------
# Embedding matrix builder — used by faiss_store.py to build the index
# ---------------------------------------------------------------------------

def build_embedding_matrix(chunks: List[Chunk]) -> Optional[np.ndarray]:
    """
    Stack all chunk embeddings into a single float32 matrix of shape
    (num_chunks, EMBEDDING_DIMENSIONS) for FAISS index construction.

    Chunks without embeddings are excluded with a warning.

    Args:
        chunks: List of embedded Chunk objects.

    Returns:
        np.ndarray of shape (n, 1536) or None if no embeddings are available.
    """
    valid = [c for c in chunks if c.embedding is not None]

    if not valid:
        logger.error("[embedder] No chunks have embeddings. Cannot build matrix.")
        return None

    if len(valid) < len(chunks):
        logger.warning(
            f"[embedder] {len(chunks) - len(valid)} chunks have no embedding "
            f"and will be excluded from the FAISS index."
        )

    matrix = np.array(
        [c.embedding for c in valid],
        dtype=np.float32,
    )

    logger.info(
        f"[embedder] Embedding matrix built: "
        f"{matrix.shape[0]} chunks × {matrix.shape[1]} dimensions."
    )

    return matrix