"""
financevault/ingestion/chunking/__init__.py

Public API for the chunking subpackage.

The pipeline and any external code should import from here only.
Internal module structure can change without breaking imports.

Usage:
    from financevault.ingestion.chunking import select_chunks_batch, StrategyConfig
"""

from .selector import select_chunks, select_chunks_batch
from .strategies import StrategyConfig, ChunkingStrategy
from .scorer import score, apply_weights, NARRATIVE_WEIGHTS, FINANCIAL_TABLE_WEIGHTS, MIXED_WEIGHTS

__all__ = [
    # Primary entry points
    "select_chunks",
    "select_chunks_batch",

    # Configuration
    "StrategyConfig",

    # Enum (re-exported for convenience)
    "ChunkingStrategy",

    # Scoring utilities (used by evaluation/)
    "score",
    "apply_weights",
    "NARRATIVE_WEIGHTS",
    "FINANCIAL_TABLE_WEIGHTS",
    "MIXED_WEIGHTS",
]