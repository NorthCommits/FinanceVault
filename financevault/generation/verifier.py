"""
financevault/generation/verifier.py

Validates, enriches, and packages the raw GPT-4o response into a
fully cited, confidence-scored, audit-trailed VerifiedResponse.

Responsibilities:
    1. Citation validation — every [Source N] in the answer must map
       to a real source. Invalid citations are stripped.
    2. Source citation building — construct SourceCitation objects
       for each referenced source with full provenance metadata.
    3. Confidence scoring — score based on retrieval signal strength,
       source corroboration, and presence of hedging language.
    4. Audit trail — immutable record of everything that happened:
       query, timestamp, chunks used, scores, strategies, confidence.

Output: VerifiedResponse — the final object returned to the app and user.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..retrieval.hybrid_retriever import RetrievalResult
from .generator import GenerationResult, QueryType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Citation pattern — matches [Source N] or [Source N][Source M] etc.
# ---------------------------------------------------------------------------
_CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]", re.IGNORECASE)

# Hedging language that lowers confidence
_HEDGING_PATTERN = re.compile(
    r"\b(approximately|roughly|estimated|may|might|could|unclear|"
    r"uncertain|not specified|not available|limited information|"
    r"insufficient data|cannot confirm|unclear from)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

@dataclass
class SourceCitation:
    """
    Full provenance record for one source used in the answer.
    Shown to the user alongside the answer for transparency.
    """
    source_num          : int
    chunk_id            : str
    company_name        : str
    ticker              : str
    fiscal_year         : int
    item_number         : str
    item_title          : str
    section_type        : str
    is_table_chunk      : bool
    chunking_strategy   : str
    chunking_score      : float
    rrf_score           : float
    cross_encoder_score : Optional[float]
    text_preview        : str        # First 200 chars of the chunk
    source_url          : Optional[str]

    def to_dict(self) -> dict:
        return {
            "source_num"          : self.source_num,
            "chunk_id"            : self.chunk_id,
            "company_name"        : self.company_name,
            "ticker"              : self.ticker,
            "fiscal_year"         : self.fiscal_year,
            "item_number"         : self.item_number,
            "item_title"          : self.item_title,
            "section_type"        : self.section_type,
            "is_table_chunk"      : self.is_table_chunk,
            "chunking_strategy"   : self.chunking_strategy,
            "chunking_score"      : self.chunking_score,
            "rrf_score"           : self.rrf_score,
            "cross_encoder_score" : self.cross_encoder_score,
            "text_preview"        : self.text_preview,
            "source_url"          : self.source_url,
        }


@dataclass
class AuditTrail:
    """
    Immutable record of everything that happened for one query.
    Used for compliance, debugging, and offline evaluation.
    """
    query_id            : str         # ISO timestamp + hash
    timestamp_utc       : str
    query               : str
    query_type          : str
    model_used          : str
    tokens_used         : int
    chunks_retrieved    : int         # From hybrid retriever
    chunks_reranked     : int         # Passed to cross-encoder
    chunks_used         : int         # In final context (top-5)
    sources_cited       : List[int]   # [Source N] numbers in the answer
    confidence          : float
    retrieval_scores    : List[dict]  # RRF + CE scores per chunk
    chunking_strategies : List[str]   # Strategy used for each chunk
    invalid_citations   : List[int]   # [Source N] that were stripped

    def to_dict(self) -> dict:
        return {
            "query_id"            : self.query_id,
            "timestamp_utc"       : self.timestamp_utc,
            "query"               : self.query,
            "query_type"          : self.query_type,
            "model_used"          : self.model_used,
            "tokens_used"         : self.tokens_used,
            "chunks_retrieved"    : self.chunks_retrieved,
            "chunks_reranked"     : self.chunks_reranked,
            "chunks_used"         : self.chunks_used,
            "sources_cited"       : self.sources_cited,
            "confidence"          : self.confidence,
            "retrieval_scores"    : self.retrieval_scores,
            "chunking_strategies" : self.chunking_strategies,
            "invalid_citations"   : self.invalid_citations,
        }


@dataclass
class VerifiedResponse:
    """
    The final output of the FinanceVault pipeline.
    This is what the app layer and the user receive.

    answer is the cleaned, citation-validated GPT response.
    sources contains full provenance for every cited source.
    confidence is a [0, 1] score reflecting answer reliability.
    audit_trail is the complete record for compliance and debugging.
    """
    query       : str
    answer      : str
    sources     : List[SourceCitation]
    confidence  : float
    query_type  : str
    audit_trail : AuditTrail
    tokens_used : int
    model       : str

    def to_dict(self) -> dict:
        return {
            "query"      : self.query,
            "answer"     : self.answer,
            "sources"    : [s.to_dict() for s in self.sources],
            "confidence" : self.confidence,
            "query_type" : self.query_type,
            "audit_trail": self.audit_trail.to_dict(),
            "tokens_used": self.tokens_used,
            "model"      : self.model,
        }

    def display(self) -> str:
        """Human-readable formatted output for CLI / notebook display."""
        lines = [
            f"QUERY: {self.query}",
            f"TYPE:  {self.query_type.upper()} | CONFIDENCE: {self.confidence:.0%} | TOKENS: {self.tokens_used}",
            "",
            "ANSWER:",
            self.answer,
            "",
            "SOURCES:",
        ]
        for s in self.sources:
            lines.append(
                f"  [Source {s.source_num}] {s.company_name} "
                f"| FY{s.fiscal_year} | Item {s.item_number} — {s.item_title}"
                + (" [TABLE]" if s.is_table_chunk else "")
            )
            lines.append(f"    {s.text_preview[:120]}...")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Citation validation
# ---------------------------------------------------------------------------

def _extract_cited_source_nums(answer: str) -> List[int]:
    """Extract all [Source N] numbers found in the answer text."""
    return [int(m.group(1)) for m in _CITATION_PATTERN.finditer(answer)]


def _validate_citations(
    answer        : str,
    valid_sources : List[int],
) -> tuple[str, List[int], List[int]]:
    """
    Validate all [Source N] citations in the answer.

    For each [Source N]:
        - If N is in valid_sources → keep it
        - If N is not in valid_sources → strip it (hallucinated reference)

    Returns:
        (cleaned_answer, valid_cited_nums, invalid_cited_nums)
    """
    cited        = _extract_cited_source_nums(answer)
    valid_set    = set(valid_sources)
    invalid_nums = [n for n in cited if n not in valid_set]
    valid_cited  = [n for n in cited if n in valid_set]

    cleaned = answer
    for invalid_num in invalid_nums:
        # Strip the invalid citation from the answer
        cleaned = re.sub(
            rf"\[Source\s+{invalid_num}\]",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        logger.warning(
            f"[verifier] Stripped invalid citation [Source {invalid_num}]. "
            f"Valid sources: {valid_sources}"
        )

    # Clean up any double spaces left after stripping
    cleaned = re.sub(r"  +", " ", cleaned).strip()

    return cleaned, list(set(valid_cited)), invalid_nums


# ---------------------------------------------------------------------------
# Source citation building
# ---------------------------------------------------------------------------

def _build_source_citations(
    results     : List[RetrievalResult],
    cited_nums  : List[int],
) -> List[SourceCitation]:
    """
    Build SourceCitation objects for each source actually cited in the answer.

    Only sources that appear in the cleaned answer are included.
    Sources retrieved but not cited are excluded from the output
    (they still appear in the audit trail).
    """
    citations: List[SourceCitation] = []
    cited_set = set(cited_nums)

    for i, result in enumerate(results, 1):
        if i not in cited_set:
            continue

        chunk = result.chunk
        meta  = chunk.metadata

        citations.append(
            SourceCitation(
                source_num          = i,
                chunk_id            = chunk.chunk_id,
                company_name        = meta.company_name,
                ticker              = meta.ticker,
                fiscal_year         = meta.fiscal_year,
                item_number         = chunk.item_number,
                item_title          = chunk.item_title,
                section_type        = chunk.section_type.value,
                is_table_chunk      = chunk.is_table_chunk,
                chunking_strategy   = chunk.chunking_strategy.value,
                chunking_score      = chunk.chunking_score,
                rrf_score           = result.rrf_score,
                cross_encoder_score = getattr(result, "cross_encoder_score", None),
                text_preview        = chunk.text[:200],
                source_url          = chunk.source_url,
            )
        )

    return citations


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def _score_confidence(
    results     : List[RetrievalResult],
    cited_nums  : List[int],
    answer      : str,
) -> float:
    """
    Compute a confidence score for the generated answer in [0, 1].

    Three components with fixed weights:

    1. Retrieval strength (0.50 weight):
       Mean cross-encoder score of cited sources (or RRF if CE unavailable).
       High retrieval scores mean the chunks are genuinely relevant.

    2. Corroboration (0.30 weight):
       Fraction of the top-5 sources that were actually cited.
       More citations = more corroboration = higher confidence.

    3. Hedging penalty (0.20 weight):
       1.0 - (hedging_word_count / total_words * 10, capped at 1.0).
       Answers full of "approximately", "may", "estimated" are less confident.
    """
    cited_set = set(cited_nums)

    # Component 1: retrieval strength
    cited_results = [r for i, r in enumerate(results, 1) if i in cited_set]

    if cited_results:
        ce_scores = [
            getattr(r, "cross_encoder_score", None)
            for r in cited_results
        ]
        if any(s is not None for s in ce_scores):
            valid_ce = [s for s in ce_scores if s is not None]
            retrieval_strength = sum(valid_ce) / len(valid_ce)
        else:
            # Fall back to normalised RRF scores
            rrf_vals = [r.rrf_score for r in cited_results]
            max_rrf  = max(rrf_vals) if rrf_vals else 1.0
            retrieval_strength = sum(rrf_vals) / (len(rrf_vals) * max_rrf) if max_rrf > 0 else 0.5
    else:
        retrieval_strength = 0.3   # Low confidence if nothing was cited

    # Component 2: corroboration
    corroboration = len(cited_nums) / len(results) if results else 0.0

    # Component 3: hedging penalty
    words         = answer.split()
    hedging_count = len(_HEDGING_PATTERN.findall(answer))
    hedging_ratio = min(1.0, hedging_count / max(len(words), 1) * 10)
    hedging_score = 1.0 - hedging_ratio

    confidence = (
        retrieval_strength * 0.50 +
        corroboration      * 0.30 +
        hedging_score      * 0.20
    )

    return round(max(0.0, min(1.0, confidence)), 4)


# ---------------------------------------------------------------------------
# Audit trail building
# ---------------------------------------------------------------------------

def _build_audit_trail(
    gen_result      : GenerationResult,
    cited_nums      : List[int],
    invalid_nums    : List[int],
    confidence      : float,
) -> AuditTrail:
    """Build the complete audit trail for this query."""
    now        = datetime.now(timezone.utc)
    query_id   = f"{now.strftime('%Y%m%dT%H%M%S')}_{abs(hash(gen_result.query)) % 10000:04d}"

    retrieval_scores = [
        {
            "source_num"          : i,
            "chunk_id"            : r.chunk.chunk_id,
            "rrf_score"           : r.rrf_score,
            "bm25_rank"           : r.bm25_rank,
            "faiss_rank"          : r.faiss_rank,
            "cross_encoder_score" : getattr(r, "cross_encoder_score", None),
            "chunking_score"      : r.chunk.chunking_score,
        }
        for i, r in enumerate(gen_result.results_used, 1)
    ]

    chunking_strategies = [
        r.chunk.chunking_strategy.value
        for r in gen_result.results_used
    ]

    return AuditTrail(
        query_id            = query_id,
        timestamp_utc       = now.isoformat(),
        query               = gen_result.query,
        query_type          = gen_result.query_type.value,
        model_used          = gen_result.model,
        tokens_used         = gen_result.tokens_used,
        chunks_retrieved    = len(gen_result.results_used),
        chunks_reranked     = len(gen_result.results_used),
        chunks_used         = len(gen_result.results_used),
        sources_cited       = cited_nums,
        confidence          = confidence,
        retrieval_scores    = retrieval_scores,
        chunking_strategies = chunking_strategies,
        invalid_citations   = invalid_nums,
    )


# ---------------------------------------------------------------------------
# Public API: verify()
# ---------------------------------------------------------------------------

def verify(gen_result: GenerationResult) -> VerifiedResponse:
    """
    Validate and enrich a GenerationResult into a VerifiedResponse.

    Steps:
        1. Validate citations — strip any [Source N] that do not exist
        2. Build SourceCitation objects for each valid cited source
        3. Score confidence based on retrieval strength + corroboration + hedging
        4. Build audit trail
        5. Return VerifiedResponse

    Args:
        gen_result: Output of generator.generate()

    Returns:
        VerifiedResponse — the final output of the FinanceVault pipeline.

    Example:
        gen    = generate(query, reranked_results, client)
        result = verify(gen)
        print(result.display())
    """
    # Handle empty/error responses gracefully
    if not gen_result.results_used:
        return VerifiedResponse(
            query       = gen_result.query,
            answer      = gen_result.raw_answer,
            sources     = [],
            confidence  = 0.0,
            query_type  = gen_result.query_type.value,
            audit_trail = AuditTrail(
                query_id            = "no_results",
                timestamp_utc       = datetime.now(timezone.utc).isoformat(),
                query               = gen_result.query,
                query_type          = gen_result.query_type.value,
                model_used          = gen_result.model,
                tokens_used         = gen_result.tokens_used,
                chunks_retrieved    = 0,
                chunks_reranked     = 0,
                chunks_used         = 0,
                sources_cited       = [],
                confidence          = 0.0,
                retrieval_scores    = [],
                chunking_strategies = [],
                invalid_citations   = [],
            ),
            tokens_used = gen_result.tokens_used,
            model       = gen_result.model,
        )

    # Step 1: Validate citations
    cleaned_answer, valid_cited, invalid_nums = _validate_citations(
        answer        = gen_result.raw_answer,
        valid_sources = gen_result.source_numbers,
    )

    if invalid_nums:
        logger.warning(
            f"[verifier] {len(invalid_nums)} invalid citation(s) stripped: "
            f"{invalid_nums}"
        )

    # Step 2: Build source citations
    sources = _build_source_citations(
        results    = gen_result.results_used,
        cited_nums = valid_cited,
    )

    # Step 3: Score confidence
    confidence = _score_confidence(
        results    = gen_result.results_used,
        cited_nums = valid_cited,
        answer     = cleaned_answer,
    )

    # Step 4: Build audit trail
    audit_trail = _build_audit_trail(
        gen_result   = gen_result,
        cited_nums   = valid_cited,
        invalid_nums = invalid_nums,
        confidence   = confidence,
    )

    logger.info(
        f"[verifier] Verification complete. "
        f"Confidence: {confidence:.0%} | "
        f"Sources cited: {len(sources)}/{len(gen_result.results_used)} | "
        f"Invalid citations stripped: {len(invalid_nums)}"
    )

    return VerifiedResponse(
        query       = gen_result.query,
        answer      = cleaned_answer,
        sources     = sources,
        confidence  = confidence,
        query_type  = gen_result.query_type.value,
        audit_trail = audit_trail,
        tokens_used = gen_result.tokens_used,
        model       = gen_result.model,
    )