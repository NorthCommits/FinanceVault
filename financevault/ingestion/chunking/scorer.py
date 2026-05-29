"""
financevault/ingestion/chunking/scorer.py

Computes the 8-metric quality scorecard for a given (strategy, chunks) pair.

Public API:
    score(chunks, section, embedder) -> MetricScores

The embedder argument is a SentenceTransformer instance passed in from
selector.py so we load the model once per run, not once per strategy call.

Metric reference:
    SC   — Size Compliance          (arithmetic, no embeddings)
    ICC  — Intrachunk Cohesion      (embeddings: sentence vs chunk)
    DCC  — Contextual Coherence     (embeddings: chunk vs neighbours)
    BI   — Block Integrity          (text matching, no embeddings)
    RC   — Missing Reference Error  (regex coreference, no embeddings)
    NDS  — Numerical Density Score  (arithmetic, no embeddings)
    TBI  — Table Boundary Integrity (text matching, no embeddings)
    SPS  — Section Purity Score     (regex item headers, no embeddings)

All scores are in [0.0, 1.0]. Higher is always better.
A score of 1.0 means the metric is perfectly satisfied.
A score of 0.0 means total failure on that metric.

Design decisions:
    - Embeddings use sentence-transformers all-MiniLM-L6-v2 (local, fast, free)
    - Each metric is a standalone function for testability
    - No metric raises — all failures return a safe default (usually 1.0 or 0.5)
    - Logging at DEBUG level for per-metric diagnostics
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import numpy as np
import tiktoken

from ..models import MetricScores, ParsedSection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Returns 0.0 if either is zero."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Sentence splitter (lightweight, no spaCy dependency)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using punctuation boundaries.
    We avoid spaCy here to keep the scorer dependency-light.
    Good enough for financial prose where sentences end with . ? !
    """
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip() and _count_tokens(s) > 3]


# ---------------------------------------------------------------------------
# Metric 1: Size Compliance (SC)
# ---------------------------------------------------------------------------

def _score_sc(
    chunks    : List[str],
    min_tokens: int = 80,
    max_tokens: int = 1200,
) -> float:
    """
    Fraction of chunks whose token count falls within [min_tokens, max_tokens].

    A chunk that is too small loses context; one that is too large overflows
    the attention budget for retrieval. Both extremes penalise this score.
    """
    if not chunks:
        return 0.0
    compliant = sum(
        1 for c in chunks
        if min_tokens <= _count_tokens(c) <= max_tokens
    )
    return round(compliant / len(chunks), 4)


# ---------------------------------------------------------------------------
# Metric 2: Intrachunk Cohesion (ICC)
# ---------------------------------------------------------------------------

def _score_icc(chunks: List[str], embedder) -> float:
    """
    Mean cosine similarity between each sentence and its parent chunk embedding.

    For each chunk:
        1. Embed the full chunk text → chunk_vec
        2. Split chunk into sentences, embed each → sent_vecs
        3. Compute cosine(sent_vec, chunk_vec) for each sentence
        4. Average across sentences → chunk_icc

    Final score = mean chunk_icc across all chunks.

    High ICC means each chunk is internally coherent — sentences all talk
    about the same financial concept. Low ICC suggests the chunk mixes topics.
    """
    if not chunks or embedder is None:
        return 1.0  # Safe default when embedder unavailable

    chunk_scores: List[float] = []

    try:
        # Batch embed all chunks at once for efficiency
        chunk_vecs = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)

        for i, chunk in enumerate(chunks):
            sentences = _split_sentences(chunk)
            if len(sentences) < 2:
                # Single-sentence chunk: perfect cohesion by definition
                chunk_scores.append(1.0)
                continue

            sent_vecs = embedder.encode(
                sentences,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            sims = [_cosine(sv, chunk_vecs[i]) for sv in sent_vecs]
            chunk_scores.append(float(np.mean(sims)))

    except Exception as e:
        logger.warning(f"[ICC] Embedding failed: {e}. Returning default 0.75.")
        return 0.75

    score = float(np.mean(chunk_scores)) if chunk_scores else 1.0
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Metric 3: Contextual Coherence (DCC)
# ---------------------------------------------------------------------------

def _score_dcc(chunks: List[str], embedder) -> float:
    """
    Mean cosine similarity between adjacent chunk pairs.

    For each consecutive pair (chunk_i, chunk_{i+1}):
        cosine(embed(chunk_i), embed(chunk_{i+1}))

    Final score = mean across all pairs.

    High DCC means chunks flow logically into each other, preserving
    the narrative thread. Low DCC means the chunking broke a coherent
    discussion across unrelated segments.

    For sections with only one chunk, DCC = 1.0 (trivially coherent).
    """
    if not chunks or embedder is None:
        return 1.0
    if len(chunks) == 1:
        return 1.0

    try:
        vecs = embedder.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
        sims = [
            _cosine(vecs[i], vecs[i + 1])
            for i in range(len(vecs) - 1)
        ]
        score = float(np.mean(sims)) if sims else 1.0
    except Exception as e:
        logger.warning(f"[DCC] Embedding failed: {e}. Returning default 0.75.")
        return 0.75

    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Metric 4: Block Integrity (BI)
# ---------------------------------------------------------------------------

# Structural block detectors
_PARAGRAPH_SPLIT  = re.compile(r"\n{2,}")
_LIST_ITEM        = re.compile(r"^\s*[-*•]\s+\S|^\s*\d+\.\s+\S", re.MULTILINE)
_TABLE_ROW        = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
_NUMBERED_HEADER  = re.compile(r"^\s*(?:Item\s+\d+[A-Z]?|ITEM\s+\d+[A-Z]?)", re.MULTILINE)


def _extract_structural_blocks(text: str) -> List[str]:
    """
    Extract structural blocks from section text.
    We collect: paragraphs, list items, table rows, and numbered headers.
    Each is a string that should appear intact within a single chunk.
    """
    blocks: List[str] = []

    # Paragraphs — split on double newlines, keep substantial ones
    for para in _PARAGRAPH_SPLIT.split(text):
        para = para.strip()
        if para and _count_tokens(para) >= 20:
            blocks.append(para)

    # List items — each item is an atomic block
    for match in _LIST_ITEM.finditer(text):
        line = match.group(0).strip()
        if line and line not in blocks:
            blocks.append(line)

    # Table rows — each row is atomic
    for match in _TABLE_ROW.finditer(text):
        row = match.group(0).strip()
        if row and row not in blocks:
            blocks.append(row)

    # Numbered headers
    for match in _NUMBERED_HEADER.finditer(text):
        header = match.group(0).strip()
        if header and header not in blocks:
            blocks.append(header)

    return blocks


def _score_bi(chunks: List[str], section: ParsedSection) -> float:
    """
    Fraction of structural blocks that appear intact in at least one chunk.

    For each block extracted from the original section text:
        - Check if any chunk contains the block as a substring
        - If yes: intact (score 1 for this block)
        - If no:  split across chunk boundaries (score 0 for this block)

    Final score = intact_blocks / total_blocks.
    Returns 1.0 if no structural blocks are detected (no penalty for empty sections).
    """
    if not chunks:
        return 0.0

    blocks = _extract_structural_blocks(section.text)
    if not blocks:
        return 1.0

    # Build a joined search space for fast membership testing
    chunk_set = chunks  # we do substring search per chunk

    intact = 0
    for block in blocks:
        # Normalise whitespace for comparison
        norm_block = " ".join(block.split())
        found = any(
            norm_block in " ".join(chunk.split())
            for chunk in chunk_set
        )
        if found:
            intact += 1

    score = intact / len(blocks)
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Metric 5: Missing Reference Error / Coreference Completeness (RC)
# ---------------------------------------------------------------------------

# Pronouns that commonly reference earlier financial entities
_PRONOUNS = re.compile(
    r"\b(it|its|they|their|them|this|these|those|such|the company|the firm|the group)\b",
    re.IGNORECASE,
)

# Financial entity patterns: company names, product names, metric labels
_ENTITY = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"    # ProperCase words
    r"|[A-Z]{2,}"                             # Acronyms: EBITDA, IPO
    r"|\$[\d,.]+[BMK]?"                       # Dollar amounts
    r"|\d+\.?\d*\s*%"                         # Percentages
    r")\b"
)


def _find_entity_pronoun_pairs(text: str) -> List[Tuple[int, int]]:
    """
    Find (entity_char_pos, pronoun_char_pos) pairs where a pronoun
    appears within 200 characters after an entity mention.
    Returns list of (entity_pos, pronoun_pos) tuples.
    """
    entities = [(m.start(), m.end()) for m in _ENTITY.finditer(text)]
    pronouns = [(m.start(), m.end()) for m in _PRONOUNS.finditer(text)]

    pairs: List[Tuple[int, int]] = []
    for e_start, e_end in entities:
        for p_start, _ in pronouns:
            if e_end <= p_start <= e_end + 200:
                pairs.append((e_start, p_start))

    return pairs


def _char_to_chunk_index(char_pos: int, chunks: List[str]) -> int:
    """Map a character position in the original text to a chunk index."""
    offset = 0
    for i, chunk in enumerate(chunks):
        if offset <= char_pos < offset + len(chunk):
            return i
        offset += len(chunk) + 1  # +1 for the separator
    return len(chunks) - 1  # Default to last chunk


def _score_rc(chunks: List[str], section: ParsedSection) -> float:
    """
    Fraction of entity-pronoun coreference pairs that are NOT broken
    across chunk boundaries.

    A pair is broken if the entity and its pronoun appear in different chunks.
    Broken pairs harm retrieval because a chunk mentioning "it declined 12%"
    is meaningless without the antecedent ("revenue") from the previous chunk.

    Returns 1.0 if no pairs are found (no penalty for clean text).
    """
    if not chunks or len(chunks) == 1:
        return 1.0

    try:
        pairs = _find_entity_pronoun_pairs(section.text)
        if not pairs:
            return 1.0

        broken = 0
        for entity_pos, pronoun_pos in pairs:
            entity_chunk  = _char_to_chunk_index(entity_pos,  chunks)
            pronoun_chunk = _char_to_chunk_index(pronoun_pos, chunks)
            if entity_chunk != pronoun_chunk:
                broken += 1

        score = 1.0 - (broken / len(pairs))
        return round(max(0.0, min(1.0, score)), 4)

    except Exception as e:
        logger.warning(f"[RC] Coreference scoring failed: {e}. Returning default 1.0.")
        return 1.0


# ---------------------------------------------------------------------------
# Metric 6: Numerical Density Score (NDS) — FinanceVault custom
# ---------------------------------------------------------------------------

_NUMERICAL_PATTERN = re.compile(
    r"""
    \$[\d,]+          |   # Dollar amounts
    \d+\.?\d*\%       |   # Percentages
    [\d,]+\s?bps      |   # Basis points
    \d[\d,]*\.\d+     |   # Decimal numbers
    \b\d{4}\b         |   # Years
    \b\d[\d,]{2,}\b       # Large integers
    """,
    re.VERBOSE,
)


def _numerical_density(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if _NUMERICAL_PATTERN.search(t))
    return hits / len(tokens)


def _score_nds(chunks: List[str], section: ParsedSection) -> float:
    """
    Measures how evenly numerical content is distributed across chunks.

    A good chunking keeps numbers near their context. A bad chunking
    dumps all numbers into one chunk and leaves others empty of data.

    Algorithm:
        1. Compute target_density = section's overall numerical_density
        2. For each chunk, compute its own numerical_density
        3. Score each chunk: 1.0 if within ±0.15 of target, else scaled distance
        4. Return mean across chunks

    For sections with very low numerical density (< 0.02), all chunks
    score 1.0 — no numbers to distribute, so no penalty.
    """
    if not chunks:
        return 0.0

    target = section.signals.numerical_density

    # Non-numerical section: NDS is trivially perfect
    if target < 0.02:
        return 1.0

    tolerance = 0.15
    chunk_scores: List[float] = []

    for chunk in chunks:
        chunk_density = _numerical_density(chunk)
        diff = abs(chunk_density - target)
        if diff <= tolerance:
            chunk_scores.append(1.0)
        else:
            # Linearly decay: at diff=0.30 → score=0.5, at diff≥0.45 → score=0.0
            decay = max(0.0, 1.0 - ((diff - tolerance) / tolerance))
            chunk_scores.append(decay)

    score = float(np.mean(chunk_scores)) if chunk_scores else 1.0
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Metric 7: Table Boundary Integrity (TBI) — FinanceVault custom
# ---------------------------------------------------------------------------

_TABLE_ROW_PATTERN = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
_TABLE_SEP_PATTERN = re.compile(r"^\s*\|[-:| ]+\|\s*$", re.MULTILINE)


def _score_tbi(chunks: List[str], section: ParsedSection) -> float:
    """
    Fraction of table rows from the original section that appear intact
    within a single chunk.

    This is our most important metric for Item 8 Financial Statements.
    A table row split across two chunks is completely unusable:
        Chunk A: "| Revenue      | $383,285 |"
        Chunk B: "| Net Income   |"   ← broken row

    Algorithm:
        1. Extract all table rows from section.text
        2. For each row, check whether any single chunk contains it verbatim
        3. Score = intact_rows / total_rows

    Returns 1.0 for sections with no tables (no penalty).
    Separator rows (|---|---| lines) are excluded — they carry no financial data.
    """
    if not chunks:
        return 0.0

    # Extract all table rows, excluding separator lines
    all_rows = [
        m.group(0).strip()
        for m in _TABLE_ROW_PATTERN.finditer(section.text)
        if not _TABLE_SEP_PATTERN.match(m.group(0))
    ]

    if not all_rows:
        return 1.0  # No tables present — perfect score by definition

    intact = 0
    for row in all_rows:
        norm_row = " ".join(row.split())
        if any(norm_row in " ".join(chunk.split()) for chunk in chunks):
            intact += 1

    score = intact / len(all_rows)
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Metric 8: Section Purity Score (SPS) — FinanceVault custom
# ---------------------------------------------------------------------------

# Detects Item headers that indicate a new 10-K section has started
_ITEM_HEADER = re.compile(
    r"(?:^|\n)\s*(?:ITEM|Item)\s+\d+[A-Za-z]?\b",
    re.MULTILINE,
)


def _score_sps(chunks: List[str], section: ParsedSection) -> float:
    """
    Fraction of chunks that do not contain an Item header from a
    different section than the one being chunked.

    This catches cases where chunking bleeds across 10-K Item boundaries.
    For example, if an Item 7 chunk also contains "Item 8. Financial Statements"
    header text, the chunk is impure and will confuse retrieval.

    Algorithm:
        1. Identify which Item this section belongs to: section.item_number
        2. For each chunk: scan for Item header patterns
        3. If a chunk contains a header for a DIFFERENT item → impure
        4. Score = pure_chunks / total_chunks

    Returns 1.0 if no Item headers are found in any chunk (pure by default).
    """
    if not chunks:
        return 0.0

    own_item = section.item_number.strip().upper()

    pure = 0
    for chunk in chunks:
        headers_in_chunk = _ITEM_HEADER.findall(chunk)
        if not headers_in_chunk:
            pure += 1
            continue

        # Check if all found headers refer to our own item number
        all_own = True
        for header in headers_in_chunk:
            # Extract the item number from the header text
            match = re.search(r"\d+[A-Za-z]?", header)
            if match:
                found_item = match.group(0).upper()
                if found_item != own_item:
                    all_own = False
                    break

        if all_own:
            pure += 1

    score = pure / len(chunks)
    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Public API: score()
# ---------------------------------------------------------------------------

def score(
    chunks  : List[str],
    section : ParsedSection,
    embedder: Optional[object] = None,
) -> MetricScores:
    """
    Compute all 8 quality metrics for a given chunking of a section.

    Args:
        chunks:   List of chunk texts produced by one strategy.
        section:  The ParsedSection that was chunked (used for reference text
                  and pre-computed signals).
        embedder: A loaded SentenceTransformer instance. If None, ICC and DCC
                  return safe defaults (0.75). Pass the same instance across
                  all strategy runs to avoid reloading the model.

    Returns:
        MetricScores with all 8 scores computed.
        weighted_total is left at 0.0 — selector.py sets it after applying
        section-type-specific weights.

    Example:
        from sentence_transformers import SentenceTransformer
        from financevault.ingestion.chunking.scorer import score

        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        scores = score(chunks, section, embedder)
        print(scores.sc, scores.icc, scores.tbi)
    """
    if not chunks:
        logger.warning(f"[scorer] Empty chunk list for {section.section_id}. Returning zero scores.")
        return MetricScores(sc=0.0, icc=0.0, dcc=0.0, bi=0.0, rc=0.0, nds=0.0, tbi=0.0, sps=0.0)

    logger.debug(f"[scorer] Scoring {len(chunks)} chunks for {section.section_id}...")

    sc  = _score_sc(chunks)
    icc = _score_icc(chunks, embedder)
    dcc = _score_dcc(chunks, embedder)
    bi  = _score_bi(chunks, section)
    rc  = _score_rc(chunks, section)
    nds = _score_nds(chunks, section)
    tbi = _score_tbi(chunks, section)
    sps = _score_sps(chunks, section)

    logger.debug(
        f"[scorer] {section.section_id} raw scores: "
        f"SC={sc:.3f} ICC={icc:.3f} DCC={dcc:.3f} BI={bi:.3f} "
        f"RC={rc:.3f} NDS={nds:.3f} TBI={tbi:.3f} SPS={sps:.3f}"
    )

    return MetricScores(
        sc  = sc,
        icc = icc,
        dcc = dcc,
        bi  = bi,
        rc  = rc,
        nds = nds,
        tbi = tbi,
        sps = sps,
        weighted_total = 0.0,  # Set by selector.py
    )


# ---------------------------------------------------------------------------
# Weight tables — used by selector.py to compute weighted_total
# These live here because they are part of the scoring definition
# ---------------------------------------------------------------------------

# Weights for NARRATIVE sections (Item 1, 1A, 1B, 2, 3, 7A, 9A)
NARRATIVE_WEIGHTS = {
    "icc": 0.25,
    "dcc": 0.20,
    "rc" : 0.20,
    "sc" : 0.15,
    "bi" : 0.10,
    "nds": 0.05,
    "tbi": 0.02,
    "sps": 0.03,
}

# Weights for FINANCIAL_TABLE sections (Item 8)
FINANCIAL_TABLE_WEIGHTS = {
    "tbi": 0.35,
    "bi" : 0.25,
    "sc" : 0.20,
    "sps": 0.10,
    "icc": 0.05,
    "dcc": 0.03,
    "rc" : 0.01,
    "nds": 0.01,
}

# Weights for MIXED sections (Item 7 MD&A)
MIXED_WEIGHTS = {
    "icc": 0.18,
    "tbi": 0.18,
    "dcc": 0.15,
    "bi" : 0.15,
    "sc" : 0.15,
    "rc" : 0.10,
    "nds": 0.05,
    "sps": 0.04,
}


def apply_weights(scores: MetricScores, weights: dict) -> float:
    """
    Compute the weighted total score from a MetricScores object
    and a weight dictionary.

    Args:
        scores:  MetricScores with all 8 individual scores set.
        weights: Dict mapping metric name to weight (must sum to 1.0).

    Returns:
        Weighted total as float in [0.0, 1.0].
    """
    total = (
        scores.sc  * weights.get("sc",  0.0) +
        scores.icc * weights.get("icc", 0.0) +
        scores.dcc * weights.get("dcc", 0.0) +
        scores.bi  * weights.get("bi",  0.0) +
        scores.rc  * weights.get("rc",  0.0) +
        scores.nds * weights.get("nds", 0.0) +
        scores.tbi * weights.get("tbi", 0.0) +
        scores.sps * weights.get("sps", 0.0)
    )
    return round(max(0.0, min(1.0, total)), 4)