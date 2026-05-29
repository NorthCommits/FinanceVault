"""
financevault/ingestion/chunking/strategies.py

The four chunking strategies available to selector.py.

Each strategy is a callable:
    strategy(section: ParsedSection, config: StrategyConfig) -> List[str]

Strategies:
    recursive_600    — RecursiveSplitter, 600-token target, tight chunks
    recursive_1100   — RecursiveSplitter, 1100-token target, broader context
    table_aware      — Line-by-line, keeps financial table rows atomic
    llm_regex        — GPT-4o generates document-specific regex split boundaries

All strategies:
    - Return List[str] of raw chunk texts (non-empty, stripped)
    - Never raise — on failure they fall back to recursive_600 and log
    - Preserve the original text content exactly (no summarisation)

Design principle:
    Strategies know nothing about scoring or selection.
    They only split. scorer.py evaluates. selector.py decides.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import tiktoken
from openai import OpenAI

from ..models import ChunkingStrategy, ParsedSection, SectionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared tokenizer — GPT-4o (cl100k_base)
# ---------------------------------------------------------------------------
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


# ---------------------------------------------------------------------------
# StrategyConfig — controls all tuneable parameters in one place
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """
    Configuration for all four strategies.
    Centralised here so selector.py can pass a single config object
    and individual strategies read only what they need.
    """
    # Recursive splitter parameters
    recursive_600_size        : int   = 600
    recursive_600_overlap     : int   = 60       # 10% overlap
    recursive_1100_size       : int   = 1100
    recursive_1100_overlap    : int   = 110      # 10% overlap
    min_chunk_tokens          : int   = 80       # Merge chunks smaller than this
    separators                : List[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )

    # Table-aware parameters
    table_min_pipe_fraction   : float = 0.5      # Fraction of chars that are | to detect table line
    table_max_tokens_per_chunk: int   = 1100     # Max tokens for a prose block between tables
    table_overlap_lines       : int   = 1        # Lines of overlap at prose/table boundaries

    # LLM regex parameters
    llm_model                 : str   = "gpt-4o"
    llm_max_input_tokens      : int   = 3000     # Truncate section before sending to GPT
    llm_temperature           : float = 0.0
    llm_min_section_tokens    : int   = 300      # Skip LLM if section is shorter than this
    llm_retry_attempts        : int   = 2

    # OpenAI client (injected by selector.py, not created here)
    openai_client             : Optional[OpenAI] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Recursive splitter — adapted from ekimetrics/adaptive-chunking
# We implement a clean version here rather than importing the library
# to avoid adding a dependency and to tailor it for financial text
# ---------------------------------------------------------------------------

class _RecursiveSplitter:
    """
    Splits text recursively using a priority list of separators.
    Merges small splits up to chunk_size with optional overlap.
    Faithfully adapted from the ekimetrics adaptive-chunking paper implementation.
    """

    def __init__(
        self,
        chunk_size    : int,
        chunk_overlap : int,
        separators    : List[str],
        min_chunk_tokens: int,
        length_fn     : Callable[[str], int] = _count_tokens,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size       = chunk_size
        self.chunk_overlap    = chunk_overlap
        self.separators       = separators
        self.min_chunk_tokens = min_chunk_tokens
        self.length_fn        = length_fn

    def _split_on_separator(self, text: str, separator: str) -> List[str]:
        """Split text on separator, keeping the separator attached to the start of the next chunk."""
        if separator == "":
            # Hard character split as last resort
            parts, step = [], max(1, self.chunk_size // 4)
            for i in range(0, len(text), step):
                parts.append(text[i : i + step])
            return [p for p in parts if p]

        pattern = re.escape(separator)
        pieces  = re.split(f"({pattern})", text)

        # Re-attach separators to the start of the following segment
        merged, i = [], 0
        while i < len(pieces):
            if i + 1 < len(pieces) and re.match(pattern, pieces[i + 1]):
                merged.append(pieces[i] + pieces[i + 1])
                i += 2
            else:
                if pieces[i]:
                    merged.append(pieces[i])
                i += 1
        return [m for m in merged if m.strip()]

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split until all pieces are within chunk_size."""
        if self.length_fn(text) <= self.chunk_size:
            return [text]

        sep        = separators[0]
        remaining  = separators[1:] if len(separators) > 1 else [""]
        splits     = self._split_on_separator(text, sep)

        result = []
        for split in splits:
            if self.length_fn(split) > self.chunk_size:
                result.extend(self._recursive_split(split, remaining))
            else:
                result.append(split)
        return result

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge small splits into chunks up to chunk_size, with overlap."""
        chunks: List[str]       = []
        current_parts: List[str]= []
        current_len             = 0

        for split in splits:
            split_len = self.length_fn(split)

            if current_len + split_len > self.chunk_size and current_parts:
                chunk = "".join(current_parts)
                if chunk.strip():
                    chunks.append(chunk.strip())

                # Build overlap from tail of current_parts
                overlap_parts, overlap_len = [], 0
                for part in reversed(current_parts):
                    part_len = self.length_fn(part)
                    if overlap_len + part_len > self.chunk_overlap:
                        break
                    overlap_parts.insert(0, part)
                    overlap_len += part_len

                current_parts = overlap_parts + [split]
                current_len   = self.length_fn("".join(current_parts))
            else:
                current_parts.append(split)
                current_len += split_len

        if current_parts:
            chunk = "".join(current_parts)
            if chunk.strip():
                chunks.append(chunk.strip())

        return chunks

    def split(self, text: str) -> List[str]:
        """Main entry point. Returns list of non-empty chunk strings."""
        if not text.strip():
            return []
        raw_splits = self._recursive_split(text.strip(), self.separators)
        merged     = self._merge_splits(raw_splits)
        # Drop any chunks that are below minimum size, merging into previous
        result: List[str] = []
        for chunk in merged:
            if result and self.length_fn(chunk) < self.min_chunk_tokens:
                result[-1] = result[-1] + "\n" + chunk
            else:
                result.append(chunk)
        return [c.strip() for c in result if c.strip()]


# ---------------------------------------------------------------------------
# Strategy 1: recursive_600
# ---------------------------------------------------------------------------

def recursive_600(section: ParsedSection, config: StrategyConfig) -> List[str]:
    """
    Recursive splitter with a 600-token target.

    Best for:
      - Risk Factors (Item 1A): dense, self-contained risk paragraphs
      - Legal Proceedings (Item 3): short, precise factual blocks
      - Controls (Item 9A): short regulatory paragraphs

    Rationale: At 600 tokens, each chunk covers roughly one complete
    financial concept or risk factor without mixing topics.
    The 10% overlap preserves sentence context at boundaries.
    """
    splitter = _RecursiveSplitter(
        chunk_size       = config.recursive_600_size,
        chunk_overlap    = config.recursive_600_overlap,
        separators       = config.separators,
        min_chunk_tokens = config.min_chunk_tokens,
    )
    try:
        chunks = splitter.split(section.text)
        logger.debug(
            f"[recursive_600] {section.section_id}: {len(chunks)} chunks "
            f"from {section.signals.token_count} tokens."
        )
        return chunks
    except Exception as e:
        logger.warning(f"[recursive_600] {section.section_id} failed: {e}. Returning full text as one chunk.")
        return [section.text.strip()]


# ---------------------------------------------------------------------------
# Strategy 2: recursive_1100
# ---------------------------------------------------------------------------

def recursive_1100(section: ParsedSection, config: StrategyConfig) -> List[str]:
    """
    Recursive splitter with a 1100-token target.

    Best for:
      - Business Description (Item 1): long product/market narratives
      - MD&A prose blocks (Item 7): multi-paragraph analysis
      - Market Risk (Item 7A): detailed quantitative discussions

    Rationale: At 1100 tokens, a chunk captures a full multi-paragraph
    discussion of one financial topic (e.g., an entire segment performance
    analysis), giving the LLM enough context to reason about it coherently.
    GPT-4o's context window handles this comfortably.
    """
    splitter = _RecursiveSplitter(
        chunk_size       = config.recursive_1100_size,
        chunk_overlap    = config.recursive_1100_overlap,
        separators       = config.separators,
        min_chunk_tokens = config.min_chunk_tokens,
    )
    try:
        chunks = splitter.split(section.text)
        logger.debug(
            f"[recursive_1100] {section.section_id}: {len(chunks)} chunks "
            f"from {section.signals.token_count} tokens."
        )
        return chunks
    except Exception as e:
        logger.warning(f"[recursive_1100] {section.section_id} failed: {e}. Returning full text as one chunk.")
        return [section.text.strip()]


# ---------------------------------------------------------------------------
# Strategy 3: table_aware
# ---------------------------------------------------------------------------

def _is_table_line(line: str, min_pipe_fraction: float) -> bool:
    """
    A line is a table line if the fraction of pipe characters
    exceeds the threshold, or if it looks like a markdown table separator.
    """
    stripped = line.strip()
    if not stripped:
        return False
    # Markdown separator row: |---|---|
    if re.match(r"^\|[-:| ]+\|$", stripped):
        return True
    pipe_count = stripped.count("|")
    if pipe_count == 0:
        return False
    return (pipe_count / len(stripped)) >= min_pipe_fraction


def _detect_table_blocks(
    lines: List[str],
    min_pipe_fraction: float,
) -> List[tuple[str, bool]]:
    """
    Group consecutive lines into (block_text, is_table) tuples.
    Consecutive table lines stay together; consecutive prose lines stay together.
    """
    if not lines:
        return []

    blocks: List[tuple[str, bool]] = []
    current_lines: List[str]       = []
    current_is_table               = _is_table_line(lines[0], min_pipe_fraction)

    for line in lines:
        line_is_table = _is_table_line(line, min_pipe_fraction)
        if line_is_table == current_is_table:
            current_lines.append(line)
        else:
            block_text = "\n".join(current_lines)
            if block_text.strip():
                blocks.append((block_text, current_is_table))
            current_lines    = [line]
            current_is_table = line_is_table

    if current_lines:
        block_text = "\n".join(current_lines)
        if block_text.strip():
            blocks.append((block_text, current_is_table))

    return blocks


def table_aware(section: ParsedSection, config: StrategyConfig) -> List[str]:
    """
    Table-preserving chunker for financial statement sections.

    Algorithm:
      1. Split the section text into consecutive (block, is_table) pairs
      2. Table blocks: kept as a single atomic chunk regardless of size
         (financial tables must never be split mid-row)
      3. Prose blocks between tables: split with recursive_1100
      4. Very large table blocks (> 2x max_tokens): split at blank table rows only
         (i.e., between separate embedded tables, not within one)

    Best for:
      - Item 8 Financial Statements: balance sheet, income statement, cash flow
      - Item 7 MD&A when table_density is high

    This is the only strategy that can achieve a high TBI (Table Boundary Integrity)
    score because it never splits a table row across chunk boundaries.
    """
    prose_splitter = _RecursiveSplitter(
        chunk_size       = config.table_max_tokens_per_chunk,
        chunk_overlap    = config.recursive_1100_overlap,
        separators       = config.separators,
        min_chunk_tokens = config.min_chunk_tokens,
    )

    lines  = section.text.splitlines()
    blocks = _detect_table_blocks(lines, config.table_min_pipe_fraction)
    chunks : List[str] = []

    for block_text, is_table in blocks:
        if not block_text.strip():
            continue

        if is_table:
            block_tokens = _count_tokens(block_text)
            if block_tokens <= config.table_max_tokens_per_chunk * 2:
                # Fits as one chunk — keep atomic
                chunks.append(block_text.strip())
            else:
                # Very large table: split at blank lines between sub-tables only
                sub_tables = re.split(r"\n{2,}", block_text)
                for sub in sub_tables:
                    if sub.strip():
                        chunks.append(sub.strip())
        else:
            # Prose block: use recursive splitter
            prose_chunks = prose_splitter.split(block_text)
            chunks.extend(prose_chunks)

    # Safety fallback: if nothing came out, return recursive_1100 result
    if not chunks:
        logger.warning(
            f"[table_aware] {section.section_id}: table detection produced no chunks. "
            f"Falling back to recursive_1100."
        )
        return recursive_1100(section, config)

    logger.debug(
        f"[table_aware] {section.section_id}: {len(chunks)} chunks "
        f"({sum(1 for b,t in blocks if t)} table blocks, "
        f"{sum(1 for b,t in blocks if not t)} prose blocks)."
    )
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Strategy 4: llm_regex
# ---------------------------------------------------------------------------

_LLM_REGEX_SYSTEM_PROMPT = """\
You are a financial document analysis expert specialising in SEC 10-K filings.
Your task: analyse the provided section text and generate ONE Python regex pattern
that identifies the most semantically meaningful split boundaries for RAG chunking.

Rules:
- The pattern must be a valid Python regex (re module compatible)
- It must match separator strings that appear BETWEEN logical chunks, not inside them
- Each resulting chunk should be self-contained and cover one financial concept
- The pattern should produce between 3 and 20 chunks for a typical section
- Do NOT split in the middle of a table row (lines containing |)
- Do NOT split in the middle of a numbered list item
- Prefer splitting at: section headers, paragraph breaks, item boundaries,
  financial statement transitions (e.g. "Net Income" → new statement section)

Return ONLY the regex pattern string. No explanation. No code fences. No preamble.
Example output:  (?=\\n(?:ITEM|Item)\\s+\\d)
"""

_LLM_REGEX_USER_TEMPLATE = """\
10-K Section: Item {item_number} — {item_title}
Section type: {section_type}
Token count: {token_count}

--- BEGIN SECTION TEXT (truncated to {max_tokens} tokens for analysis) ---
{text_sample}
--- END SECTION TEXT ---

Generate the regex split pattern for this section:
"""


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens."""
    tokens    = _TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _TOKENIZER.decode(tokens[:max_tokens])


def _validate_regex(pattern: str) -> bool:
    """Check that the pattern compiles and is not trivially wrong."""
    if not pattern or len(pattern.strip()) < 2:
        return False
    try:
        compiled = re.compile(pattern)
        # Must not match empty string (would cause infinite splits)
        if compiled.match(""):
            return False
        return True
    except re.error:
        return False


def _apply_regex_split(text: str, pattern: str) -> List[str]:
    """
    Split text on the LLM-generated regex pattern.
    Separator is attached to the start of the following chunk (like the paper).
    """
    try:
        pieces = re.split(f"({pattern})", text)
    except re.error as e:
        logger.warning(f"[llm_regex] Regex split failed ({e}), returning unsplit text.")
        return [text.strip()]

    # Re-attach separators to following segment
    chunks: List[str] = []
    i = 0
    while i < len(pieces):
        if i + 1 < len(pieces):
            segment = (pieces[i] + pieces[i + 1]).strip()
            i += 2
        else:
            segment = pieces[i].strip()
            i += 1
        if segment:
            chunks.append(segment)

    return chunks if chunks else [text.strip()]


def llm_regex(section: ParsedSection, config: StrategyConfig) -> List[str]:
    """
    GPT-4o-powered regex chunker.

    Process:
      1. If section is too short (< llm_min_section_tokens), skip and
         fall back to recursive_600 immediately (not worth the API cost)
      2. Truncate section text to llm_max_input_tokens for the API call
      3. Call GPT-4o with the system + user prompt to get a regex pattern
      4. Validate the pattern (compiles, non-trivial, produces splits)
      5. Apply the pattern to the FULL section text (not the truncated version)
      6. If anything fails at any step, fall back to recursive_600

    Best for:
      - Sections with clear structural boundaries GPT can detect
        (numbered sub-items, alternating prose/data blocks, topic transitions)
      - Long sections (> 2000 tokens) where fixed-size chunking loses structure

    Cost note: Each call uses ~500-1000 input tokens (truncated sample) + ~50 output
    tokens (the regex). At gpt-4o pricing this is < $0.01 per section.
    With 10 companies × 9 sections, total LLM regex cost ≈ $0.50-$1.00 per full run.
    """
    # Short section guard
    if section.signals.token_count < config.llm_min_section_tokens:
        logger.debug(
            f"[llm_regex] {section.section_id}: "
            f"{section.signals.token_count} tokens < minimum {config.llm_min_section_tokens}. "
            f"Falling back to recursive_600."
        )
        return recursive_600(section, config)

    # Require OpenAI client
    client = config.openai_client
    if client is None:
        logger.warning(
            f"[llm_regex] {section.section_id}: No OpenAI client provided. "
            f"Falling back to recursive_600."
        )
        return recursive_600(section, config)

    # Prepare truncated text sample for the API call
    text_sample = _truncate_to_tokens(section.text, config.llm_max_input_tokens)
    user_prompt = _LLM_REGEX_USER_TEMPLATE.format(
        item_number  = section.item_number,
        item_title   = section.item_title,
        section_type = section.section_type.value,
        token_count  = section.signals.token_count,
        max_tokens   = config.llm_max_input_tokens,
        text_sample  = text_sample,
    )

    pattern: Optional[str] = None

    for attempt in range(config.llm_retry_attempts):
        try:
            response = client.chat.completions.create(
                model       = config.llm_model,
                temperature = config.llm_temperature,
                max_tokens  = 150,
                messages    = [
                    {"role": "system", "content": _LLM_REGEX_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            raw_pattern = response.choices[0].message.content.strip()

            # Strip any accidental code fences
            raw_pattern = re.sub(r"^```[a-z]*\n?", "", raw_pattern)
            raw_pattern = re.sub(r"\n?```$",        "", raw_pattern)
            raw_pattern = raw_pattern.strip()

            if _validate_regex(raw_pattern):
                pattern = raw_pattern
                logger.debug(
                    f"[llm_regex] {section.section_id}: "
                    f"Got pattern on attempt {attempt+1}: {pattern!r}"
                )
                break
            else:
                logger.warning(
                    f"[llm_regex] {section.section_id}: "
                    f"Invalid pattern on attempt {attempt+1}: {raw_pattern!r}"
                )
                time.sleep(0.5)

        except Exception as e:
            logger.warning(
                f"[llm_regex] {section.section_id}: "
                f"API call failed on attempt {attempt+1}: {type(e).__name__}: {e}"
            )
            time.sleep(1.0)

    # If we never got a valid pattern, fall back
    if pattern is None:
        logger.warning(
            f"[llm_regex] {section.section_id}: "
            f"All {config.llm_retry_attempts} attempts failed. "
            f"Falling back to recursive_600."
        )
        return recursive_600(section, config)

    # Apply the validated pattern to the FULL section text
    raw_chunks = _apply_regex_split(section.text, pattern)

    # Post-process: merge very small chunks, split very large ones
    final_chunks: List[str] = []
    fallback_splitter = _RecursiveSplitter(
        chunk_size       = config.recursive_1100_size,
        chunk_overlap    = config.recursive_1100_overlap,
        separators       = config.separators,
        min_chunk_tokens = config.min_chunk_tokens,
    )

    for chunk in raw_chunks:
        chunk_tokens = _count_tokens(chunk)
        if chunk_tokens > config.recursive_1100_size * 2:
            # Chunk is too large — split it further with recursive_1100
            sub_chunks = fallback_splitter.split(chunk)
            final_chunks.extend(sub_chunks)
        elif chunk_tokens < config.min_chunk_tokens and final_chunks:
            # Chunk is too small — merge into previous
            final_chunks[-1] = final_chunks[-1] + "\n" + chunk
        else:
            final_chunks.append(chunk)

    final_chunks = [c.strip() for c in final_chunks if c.strip()]

    if not final_chunks:
        logger.warning(
            f"[llm_regex] {section.section_id}: "
            f"Post-processing produced no chunks. Falling back to recursive_600."
        )
        return recursive_600(section, config)

    logger.debug(
        f"[llm_regex] {section.section_id}: "
        f"{len(final_chunks)} chunks produced. Pattern: {pattern!r}"
    )

    return final_chunks


# ---------------------------------------------------------------------------
# Strategy registry — selector.py uses this to dispatch by enum
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[ChunkingStrategy, Callable] = {
    ChunkingStrategy.RECURSIVE_600  : recursive_600,
    ChunkingStrategy.RECURSIVE_1100 : recursive_1100,
    ChunkingStrategy.TABLE_AWARE    : table_aware,
    ChunkingStrategy.LLM_REGEX      : llm_regex,
}


def run_strategy(
    strategy : ChunkingStrategy,
    section  : ParsedSection,
    config   : StrategyConfig,
) -> List[str]:
    """
    Dispatch a strategy by enum value.
    Used by selector.py so it never imports individual strategy functions directly.

    Args:
        strategy: ChunkingStrategy enum value
        section:  ParsedSection to chunk
        config:   StrategyConfig with all parameters

    Returns:
        List[str] of chunk texts. Never empty — worst case returns [section.text].
    """
    fn     = STRATEGY_REGISTRY.get(strategy)
    if fn is None:
        logger.error(f"Unknown strategy: {strategy}. Falling back to recursive_600.")
        fn = recursive_600

    chunks = fn(section, config)

    # Absolute safety net: never return an empty list
    if not chunks:
        logger.warning(f"Strategy {strategy} returned no chunks for {section.section_id}. Using full text.")
        chunks = [section.text.strip()]

    return chunks