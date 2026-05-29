"""
financevault/generation/__init__.py

Public API for the generation package.

Usage:
    from financevault.generation import generate, verify
    from financevault.generation import VerifiedResponse, SourceCitation, AuditTrail
"""

from .generator import generate, classify_query, QueryType, GenerationResult
from .verifier  import verify, VerifiedResponse, SourceCitation, AuditTrail

__all__ = [
    # Core functions
    "generate",
    "verify",

    # Query classification
    "classify_query",
    "QueryType",

    # Data models
    "GenerationResult",
    "VerifiedResponse",
    "SourceCitation",
    "AuditTrail",
]