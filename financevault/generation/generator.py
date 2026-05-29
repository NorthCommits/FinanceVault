"""
financevault/generation/generator.py

Assembles context from reranked chunks and calls GPT-4o to generate
a grounded, cited answer to the user's financial query.

Responsibilities:
    1. Classify query type (factual / comparative / analytical)
    2. Assemble structured context block from top-5 RetrievalResults
    3. Select the appropriate system prompt for the query type
    4. Call GPT-4o with the assembled prompt
    5. Return raw GPT response + token usage + query type for verifier.py

What this file does NOT do:
    - Citation validation    (verifier.py)
    - Confidence scoring     (verifier.py)
    - Audit trail building   (verifier.py)

Context engineering decisions:
    - Table chunks get a [TABLE] prefix so GPT knows to treat them as data
    - Each source is labelled with company, filing year, and item number
    - System prompts are specialised per query type — factual queries get
      a brevity instruction, analytical queries get a reasoning instruction
    - GPT is explicitly told to cite every factual claim with [Source N]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from openai import OpenAI

from ..retrieval.hybrid_retriever import RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query type classification
# ---------------------------------------------------------------------------

class QueryType(str, Enum):
    FACTUAL     = "factual"      # Single fact lookup
    COMPARATIVE = "comparative"  # Compare two or more entities
    ANALYTICAL  = "analytical"   # Reasoning, risk analysis, trends


# Keywords that signal each query type
_COMPARATIVE_SIGNALS = re.compile(
    r"\b(compare|versus|vs\.?|relative to|against|difference between|"
    r"how does .+ compare|better|worse|higher|lower than)\b",
    re.IGNORECASE,
)

_ANALYTICAL_SIGNALS = re.compile(
    r"\b(why|explain|analyse|analyze|assess|evaluate|impact|implication|"
    r"risk|trend|outlook|strategy|what does .+ mean|how will|forecast|"
    r"should|recommend|concern|challenge|opportunity)\b",
    re.IGNORECASE,
)


def classify_query(query: str) -> QueryType:
    """
    Classify a query into one of three types for prompt selection.

    Priority order: comparative > analytical > factual
    If no strong signals are found, defaults to factual.
    """
    if _COMPARATIVE_SIGNALS.search(query):
        return QueryType.COMPARATIVE
    if _ANALYTICAL_SIGNALS.search(query):
        return QueryType.ANALYTICAL
    return QueryType.FACTUAL


# ---------------------------------------------------------------------------
# System prompts — one per query type
# ---------------------------------------------------------------------------

_BASE_CITATION_INSTRUCTION = """
Every factual claim in your answer MUST be followed immediately by a citation
in the format [Source N] where N matches the source number in the context.
If a claim is supported by multiple sources, cite all of them: [Source 1][Source 3].
Do not cite a source for general financial knowledge — only for specific facts
drawn from the provided context.
Never fabricate numbers, dates, or metrics. If the context does not contain
enough information to answer fully, say so explicitly.
"""

SYSTEM_PROMPTS = {
    QueryType.FACTUAL: f"""You are FinanceVault, an expert financial analyst assistant
specialising in SEC 10-K filings and annual reports.

Your task: answer the user's question accurately and concisely using ONLY
the provided source excerpts from SEC filings.

Guidelines:
- Be precise and direct. Lead with the specific number or fact being asked for.
- Keep your answer focused. Do not add context beyond what the question asks.
- Use exact figures from the sources, not approximations.
{_BASE_CITATION_INSTRUCTION}""",

    QueryType.COMPARATIVE: f"""You are FinanceVault, an expert financial analyst assistant
specialising in SEC 10-K filings and annual reports.

Your task: compare the entities, metrics, or time periods in the user's question
using ONLY the provided source excerpts from SEC filings.

Guidelines:
- Structure your comparison clearly. Use a consistent format for each entity.
- Highlight the key differences and similarities with specific numbers.
- If data for one entity is missing from the sources, state that explicitly.
- Consider absolute values AND relative metrics (margins, ratios, growth rates).
{_BASE_CITATION_INSTRUCTION}""",

    QueryType.ANALYTICAL: f"""You are FinanceVault, an expert financial analyst assistant
specialising in SEC 10-K filings and annual reports.

Your task: provide a thoughtful analytical response to the user's question
grounded in the provided source excerpts from SEC filings.

Guidelines:
- Structure your reasoning: key findings → implications → conclusion.
- Distinguish between facts from the sources and your analytical inference.
- Label inferences clearly: "This suggests...", "Based on these figures..."
- Acknowledge limitations honestly if the sources do not cover all aspects.
- Be substantive. Analytical questions deserve thorough responses.
{_BASE_CITATION_INSTRUCTION}""",
}


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def _format_source_block(result: RetrievalResult, source_num: int) -> str:
    """
    Format a single RetrievalResult as a labelled source block for the prompt.

    Format:
        [SOURCE N] {Company} | 10-K FY{Year} | Item {N} — {Title}
        [TABLE] or [TEXT]
        {chunk text}
    """
    chunk    = result.chunk
    meta     = chunk.metadata
    prefix   = "[TABLE]" if chunk.is_table_chunk else "[TEXT]"

    header = (
        f"[SOURCE {source_num}] "
        f"{meta.company_name} | "
        f"10-K FY{meta.fiscal_year} | "
        f"Item {chunk.item_number} — {chunk.item_title}"
    )

    return f"{header}\n{prefix}\n{chunk.text.strip()}"


def assemble_context(results: List[RetrievalResult]) -> tuple[str, List[int]]:
    """
    Build a structured context block from the top reranked results.

    Returns:
        (context_text, list_of_valid_source_numbers)
        The source numbers list is passed to verifier.py for citation validation.
    """
    blocks       : List[str] = []
    source_numbers: List[int] = []

    for i, result in enumerate(results, 1):
        block = _format_source_block(result, i)
        blocks.append(block)
        source_numbers.append(i)

    context = "\n\n" + ("\n\n" + "—" * 60 + "\n\n").join(blocks) + "\n\n"
    return context, source_numbers


# ---------------------------------------------------------------------------
# GenerationResult — raw output before verification
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """
    Raw output from GPT-4o before verifier.py processes it.
    Passed directly to verifier.verify().
    """
    query        : str
    raw_answer   : str
    query_type   : QueryType
    results_used : List[RetrievalResult]
    source_numbers: List[int]
    tokens_used  : int
    model        : str


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate(
    query      : str,
    results    : List[RetrievalResult],
    client     : OpenAI,
    model      : str = "gpt-4o",
    max_tokens : int = 1000,
) -> GenerationResult:
    """
    Generate a grounded, cited answer to the user's financial query.

    Args:
        query:      User's natural language question.
        results:    Top-k RetrievalResults from the reranker (typically 5).
        client:     Authenticated OpenAI client.
        model:      GPT model to use. Default gpt-4o.
        max_tokens: Maximum tokens in the GPT response.

    Returns:
        GenerationResult with raw_answer, query_type, and token usage.
        Pass this to verifier.verify() to get the final VerifiedResponse.

    Example:
        from financevault.generation import generate, verify

        gen_result = generate(query, reranked_results, openai_client)
        response   = verify(gen_result)
        print(response.answer)
    """
    if not results:
        logger.warning("[generator] No results provided. Generating refusal.")
        return GenerationResult(
            query         = query,
            raw_answer    = (
                "I could not find relevant information in the available "
                "SEC filings to answer this question. Please try rephrasing "
                "your query or check that the relevant company and year are indexed."
            ),
            query_type    = QueryType.FACTUAL,
            results_used  = [],
            source_numbers= [],
            tokens_used   = 0,
            model         = model,
        )

    # Classify query
    query_type    = classify_query(query)
    system_prompt = SYSTEM_PROMPTS[query_type]

    logger.info(
        f"[generator] Query type: {query_type.value} | "
        f"Sources: {len(results)} | Model: {model}"
    )

    # Assemble context
    context, source_numbers = assemble_context(results)

    # Build user message
    user_message = (
        f"FINANCIAL FILING EXCERPTS:\n"
        f"{context}"
        f"QUESTION: {query.strip()}\n\n"
        f"Answer using only the sources above. "
        f"Cite every factual claim with [Source N]."
    )

    # Call GPT-4o
    try:
        response = client.chat.completions.create(
            model      = model,
            max_tokens = max_tokens,
            temperature= 0.1,   # Low temperature for factual accuracy
            messages   = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )

        raw_answer  = response.choices[0].message.content.strip()
        tokens_used = response.usage.total_tokens

        logger.info(
            f"[generator] Response generated. "
            f"Tokens used: {tokens_used}. "
            f"Answer length: {len(raw_answer)} chars."
        )

        return GenerationResult(
            query         = query,
            raw_answer    = raw_answer,
            query_type    = query_type,
            results_used  = results,
            source_numbers= source_numbers,
            tokens_used   = tokens_used,
            model         = model,
        )

    except Exception as e:
        logger.error(f"[generator] GPT-4o call failed: {type(e).__name__}: {e}")
        return GenerationResult(
            query         = query,
            raw_answer    = (
                f"An error occurred while generating the response: {e}. "
                f"Please try again."
            ),
            query_type    = query_type,
            results_used  = results,
            source_numbers= source_numbers,
            tokens_used   = 0,
            model         = model,
        )