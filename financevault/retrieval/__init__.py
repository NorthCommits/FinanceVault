"""
financevault/retrieval/__init__.py

Public API for the retrieval package.

Usage:
    from financevault.retrieval import HybridRetriever, FAISSStore, BM25Store
    from financevault.retrieval import embed_chunks, embed_query, RetrievalResult
"""

from .embedder import embed_chunks, embed_query, build_embedding_matrix, EMBEDDING_DIMENSIONS
from .faiss_store import FAISSStore
from .bm25_store import BM25Store
from .hybrid_retriever import HybridRetriever, RetrievalResult

__all__ = [
    # Stores
    "FAISSStore",
    "BM25Store",

    # Retriever
    "HybridRetriever",
    "RetrievalResult",

    # Embedding utilities
    "embed_chunks",
    "embed_query",
    "build_embedding_matrix",
    "EMBEDDING_DIMENSIONS",
]