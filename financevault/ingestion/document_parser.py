"""
financevault/ingestion/document_parser.py

Converts a RawFiling (edgartools TenK object) into a list of ParsedSection models.
This is the translation layer between edgartools and our chunking pipeline.

Responsibilities:
  - Extract text for each standard 10-K section via edgartools properties
  - Serialize financial tables (balance sheet, income statement, cash flow)
    from DataFrames into markdown strings that preserve row integrity
  - Detect SectionSignals per section (numerical density, table density, etc.)
  - Classify each section as NARRATIVE, FINANCIAL_TABLE, or MIXED
  - Handle missing sections, malformed data, and edgartools edge cases gracefully
  - Return List[ParsedSection] — one entry per non-empty section found

What this file does NOT do:
  - Fetch anything from EDGAR      (that is edgar_fetcher.py)
  - Chunk any text                 (that is chunking/selector.py)
  - Embed anything                 (that is retrieval/embedder.py)

Section coverage:
  Item 1   — Business
  Item 1A  — Risk Factors
  Item 1B  — Unresolved Staff Comments  (often short/empty, included for completeness)
  Item 2   — Properties
  Item 3   — Legal Proceedings
  Item 7   — MD&A
  Item 7A  — Quantitative and Qualitative Disclosures about Market Risk
  Item 8   — Financial Statements (balance sheet + income statement + cash flow)
  Item 9A  — Controls and Procedures
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Tuple

import tiktoken

from .models import FilingMetadata, ParsedSection, RawFiling, SectionSignals, SectionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer — shared across all signal computation in this file
# We use cl100k_base which is the tokenizer for GPT-4o
# ---------------------------------------------------------------------------
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count tokens using GPT-4o tokenizer."""
    return len(_TOKENIZER.encode(text))


# ---------------------------------------------------------------------------
# Section definitions — maps our internal keys to:
#   - edgartools TenK property name
#   - human-readable item number and title
#   - default section type (may be overridden by signal detection)
# ---------------------------------------------------------------------------
_SECTION_MAP: list[dict] = [
    {
        "key"         : "business",
        "property"    : "business",
        "item_number" : "1",
        "item_title"  : "Business",
        "default_type": SectionType.NARRATIVE,
    },
    {
        "key"         : "risk_factors",
        "property"    : "risk_factors",
        "item_number" : "1A",
        "item_title"  : "Risk Factors",
        "default_type": SectionType.NARRATIVE,
    },
    {
        "key"         : "unresolved_comments",
        "property"    : "unresolved_staff_comments",
        "item_number" : "1B",
        "item_title"  : "Unresolved Staff Comments",
        "default_type": SectionType.NARRATIVE,
    },
    {
        "key"         : "properties",
        "property"    : "properties",
        "item_number" : "2",
        "item_title"  : "Properties",
        "default_type": SectionType.NARRATIVE,
    },
    {
        "key"         : "legal_proceedings",
        "property"    : "legal_proceedings",
        "item_number" : "3",
        "item_title"  : "Legal Proceedings",
        "default_type": SectionType.NARRATIVE,
    },
    {
        "key"         : "mda",
        "property"    : "management_discussion",
        "item_number" : "7",
        "item_title"  : "Management's Discussion and Analysis",
        "default_type": SectionType.MIXED,
    },
    {
        "key"         : "market_risk",
        "property"    : "market_risk",
        "item_number" : "7A",
        "item_title"  : "Quantitative and Qualitative Disclosures about Market Risk",
        "default_type": SectionType.NARRATIVE,
    },
    {
        "key"         : "controls",
        "property"    : "controls_and_procedures",
        "item_number" : "9A",
        "item_title"  : "Controls and Procedures",
        "default_type": SectionType.NARRATIVE,
    },
]

# Item 8 is handled separately because it comes from financials, not text properties
_ITEM8_DEFINITION = {
    "key"         : "financial_statements",
    "item_number" : "8",
    "item_title"  : "Financial Statements and Supplementary Data",
    "default_type": SectionType.FINANCIAL_TABLE,
}

# ---------------------------------------------------------------------------
# Numerical density helpers
# Tokens that count as "numerical" in financial text
# ---------------------------------------------------------------------------
_NUMERICAL_PATTERN = re.compile(
    r"""
    \$[\d,]+          |  # Dollar amounts:  $1,200
    \d+\.?\d*\%       |  # Percentages:     12.5%
    [\d,]+\s?bps      |  # Basis points:    25 bps
    \d[\d,]*\.\d+     |  # Decimals:        3.14
    \b\d{4}\b         |  # Years:           2024
    \b\d[\d,]{2,}\b      # Large numbers:   1,200
    """,
    re.VERBOSE,
    
)

# ---------------------------------------------------------------------------
# XBRL label cleaning + number formatting
# ---------------------------------------------------------------------------
_XBRL_PREFIX = re.compile(r"^[a-zA-Z]+[-_]")
_CAMEL_SPLIT  = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _clean_xbrl_label(raw: str) -> str:
    """Convert XBRL taxonomy string to human-readable label.
    us-gaap_NetIncomeLoss → Net Income Loss
    jpm_TotalAssets       → Total Assets
    """
    if not raw or str(raw).strip() in ("nan", "None", ""):
        return ""
    label = str(raw).strip()
    label = _XBRL_PREFIX.sub("", label)
    label = _CAMEL_SPLIT.sub(" ", label)
    label = label.replace("_", " ").replace("-", " ")
    return " ".join(label.split()).title()


def _format_number(raw) -> str:
    """Format raw XBRL numeric to human-readable financial figure.
    33680000000.0 → $33.68B
    nan           → —
    """
    import math
    try:
        val = float(raw)
    except (TypeError, ValueError):
        s = str(raw).strip()
        return s if s not in ("nan", "None", "") else "—"
    if math.isnan(val):
        return "—"
    abs_val = abs(val)
    if abs_val >= 1_000_000_000:
        return f"${val / 1_000_000_000:.2f}B"
    if abs_val >= 1_000_000:
        return f"${val / 1_000_000:.0f}M"
    if abs_val >= 1_000:
        return f"${val / 1_000:.0f}K"
    if 0 < abs_val < 2:
        return f"{val * 100:.2f}%"
    return f"{val:,.0f}"


def _compute_numerical_density(text: str) -> float:
    """
    Fraction of whitespace-delimited tokens that look numerical.
    Returns 0.0 for empty text.
    """
    if not text.strip():
        return 0.0
    tokens = text.split()
    if not tokens:
        return 0.0
    numerical_hits = sum(1 for t in tokens if _NUMERICAL_PATTERN.search(t))
    return round(numerical_hits / len(tokens), 4)


def _compute_table_density(text: str) -> float:
    """
    Fraction of lines that look like table rows.
    Table rows contain pipe characters or multiple tab/space-separated columns
    that include at least one number.
    Returns 0.0 for empty text.
    """
    if not text.strip():
        return 0.0
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    table_lines = sum(
        1 for line in lines
        if "|" in line or (
            len(line.split()) >= 3
            and _NUMERICAL_PATTERN.search(line)
        )
    )
    return round(table_lines / len(lines), 4)


def _compute_avg_sentence_length(text: str) -> float:
    """Mean token count per sentence. Sentences split on . ? !"""
    if not text.strip():
        return 0.0
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    lengths = [_count_tokens(s) for s in sentences]
    return round(sum(lengths) / len(lengths), 2)


def _detect_subsections(text: str) -> bool:
    """True if the text contains numbered sub-items like '1.', '(a)', 'i.'"""
    patterns = [
        r"^\s*\d+\.\s+[A-Z]",       # 1. Something
        r"^\s*\([a-z]\)\s+[A-Z]",   # (a) Something
        r"^\s*[ivxlc]+\.\s+[A-Z]",  # iv. Something
    ]
    for pattern in patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True
    return False


def _classify_section_type(
    default_type: SectionType,
    numerical_density: float,
    table_density: float,
) -> SectionType:
    """
    Override the default section type based on detected signals.
    A section with high table_density that was labelled NARRATIVE gets promoted to MIXED.
    A section with very high table_density becomes FINANCIAL_TABLE.
    """
    if table_density >= 0.45:
        return SectionType.FINANCIAL_TABLE
    if table_density >= 0.20 or numerical_density >= 0.15:
        if default_type == SectionType.NARRATIVE:
            return SectionType.MIXED
    return default_type


def _build_signals(text: str) -> SectionSignals:
    """Compute all SectionSignals for a given section text."""
    lines     = [l for l in text.splitlines() if l.strip()]
    paragraphs = [p for p in re.split(r"\n{2,}", text) if p.strip()]

    # Rough table row count: lines that contain | or look columnar with numbers
    table_rows = sum(
        1 for line in lines
        if "|" in line or (
            len(line.split()) >= 3
            and _NUMERICAL_PATTERN.search(line)
        )
    )

    num_density   = _compute_numerical_density(text)
    table_density = _compute_table_density(text)

    return SectionSignals(
        token_count         = _count_tokens(text),
        numerical_density   = num_density,
        table_density       = table_density,
        avg_sentence_length = _compute_avg_sentence_length(text),
        has_subsections     = _detect_subsections(text),
        table_row_count     = table_rows,
        paragraph_count     = len(paragraphs),
    )


# ---------------------------------------------------------------------------
# Table serialization
# edgartools gives us DataFrames; we convert them to markdown
# preserving row integrity for the table_aware chunking strategy
# ---------------------------------------------------------------------------

def _dataframe_to_markdown(df: Any, title: str = "") -> Optional[str]:
    """
    Convert a pandas DataFrame (financial statement) to a clean markdown table.
    XBRL taxonomy labels are cleaned to human-readable text.
    Raw numbers are formatted as $33,680M, $1.2B, 46.21% etc.
    Each row kept on a single line for Table Boundary Integrity.
    """
    try:
        if df is None or df.empty:
            return None

        lines: list[str] = []
        if title:
            lines.append(f"### {title}")
            lines.append("")

        # ---------------------------------------------------------------------------
        # Filter to only meaningful columns
        # Keep: columns that look like years/periods (2024, 2023, FY2024, 2024 12 31)
        # Drop: XBRL metadata columns (Concept, Label, Level, Abstract, Axis, etc.)
        # ---------------------------------------------------------------------------
        _XBRL_META_COLS = {
            "concept", "label", "level", "abstract", "dimension",
            "breakdown", "axis", "member", "balance", "weight",
            "sign", "member label", "abstract concept", "concept label"
        }
        _YEAR_PATTERN = re.compile(r"\d{4}")

        keep_cols = []
        for c in df.columns:
            s = str(c).strip()
            # Keep if it contains a 4-digit year
            if _YEAR_PATTERN.search(s):
                keep_cols.append(c)
            # Drop if it is a known XBRL metadata column
            elif s.lower() in _XBRL_META_COLS:
                continue
            # Drop single-word technical columns
            elif len(s.split()) <= 1 and not s[0].isdigit():
                continue
            else:
                keep_cols.append(c)

        if not keep_cols:
            return None

        # Clean column names for display
        col_names = []
        for c in keep_cols:
            s = str(c).strip()
            # Shorten date-like column names: "2024 12 31 (Fy)" → "FY2024"
            year_match = re.search(r"(\d{4})", s)
            if year_match:
                col_names.append(f"FY{year_match.group(1)}")
            else:
                col_names.append(_clean_xbrl_label(s) or s)

        # Also look for a human-readable label column
        label_col = None
        for c in df.columns:
            if str(c).strip().lower() == "label":
                label_col = c
                break

        header    = "| Metric | " + " | ".join(col_names) + " |"
        separator = "| --- | " + " | ".join(["---"] * len(col_names)) + " |"
        lines.append(header)
        lines.append(separator)

        for idx, row in df.iterrows():
            # Use Label column if available, otherwise clean the index
            if label_col and str(row.get(label_col, "")).strip() not in ("nan", "None", ""):
                label = str(row[label_col]).strip().replace("\n", " ")
            else:
                raw_label = str(idx).strip()
                label     = _clean_xbrl_label(raw_label) or raw_label
                label     = label.replace("\n", " ")

            # Get values only for kept columns
            values_raw = [str(row.get(c, "")) for c in keep_cols]

            # Skip rows where all values are empty
            if all(v.strip() in ("nan", "None", "", "—") for v in values_raw):
                continue

            # Format each value
            values = [_format_number(v).replace("\n", " ") for v in values_raw]

            data_row = f"| {label} | " + " | ".join(values) + " |"
            lines.append(data_row)

        return "\n".join(lines) if len(lines) > 2 else None

    except Exception as e:
        logger.warning(f"Failed to convert DataFrame to markdown: {e}")
        return None


def _extract_financial_tables(filing_obj: Any, ticker: str) -> Tuple[str, List[str]]:
    """
    Extract Item 8 content: balance sheet, income statement, and cash flow.
    Returns (combined_text, list_of_markdown_tables).

    Uses edgartools' XBRL-based financials API with graceful fallbacks.
    """
    tables: list[str]       = []
    text_parts: list[str]   = []

    statement_configs = [
        ("income_statement",  "Income Statement"),
        ("balance_sheet",     "Balance Sheet"),
        ("cashflow_statement","Cash Flow Statement"),
    ]

    try:
        # Primary path: XBRL-based statements via filing.xbrl()
        xbrl       = filing_obj._filing.xbrl() if hasattr(filing_obj, "_filing") else None
        statements = xbrl.statements if xbrl else None

        if statements:
            for method_name, title in statement_configs:
                try:
                    stmt = getattr(statements, method_name)()
                    df   = stmt.to_dataframe(view="standard")
                    md   = _dataframe_to_markdown(df, title=title)
                    if md:
                        tables.append(md)
                        text_parts.append(f"[{title}]\n{md}")
                        logger.debug(f"[{ticker}] Extracted {title} via XBRL statements.")
                except Exception as e:
                    logger.debug(f"[{ticker}] XBRL {method_name} failed: {e}")

    except Exception as e:
        logger.debug(f"[{ticker}] XBRL path failed: {e}. Trying Company.get_financials().")

    # Fallback path: Company.get_financials() standardized methods
    # This runs if XBRL gave us nothing at all
    if not tables:
        try:
            from edgar import Company
            company    = Company(ticker)
            financials = company.get_financials()

            if financials:
                fallback_stmts = [
                    (financials.income_statement,   "Income Statement"),
                    (financials.balance_sheet,       "Balance Sheet"),
                    (financials.cashflow_statement,  "Cash Flow Statement"),
                ]
                for stmt_attr, title in fallback_stmts:
                    try:
                        stmt = stmt_attr() if callable(stmt_attr) else stmt_attr
                        df   = stmt.to_dataframe() if hasattr(stmt, "to_dataframe") else None
                        if df is not None:
                            md = _dataframe_to_markdown(df, title=title)
                            if md:
                                tables.append(md)
                                text_parts.append(f"[{title}]\n{md}")
                                logger.debug(f"[{ticker}] Extracted {title} via get_financials() fallback.")
                    except Exception as e:
                        logger.debug(f"[{ticker}] Fallback {title} failed: {e}")

        except Exception as e:
            logger.warning(f"[{ticker}] Both XBRL and get_financials() failed for Item 8: {e}")

    combined_text = "\n\n".join(text_parts) if text_parts else ""
    return combined_text, tables


# ---------------------------------------------------------------------------
# Core parsing logic
# ---------------------------------------------------------------------------

def _safe_get_text(filing_obj: Any, property_name: str) -> Optional[str]:
    """
    Safely retrieve a text property from an edgartools TenK object.
    Returns None if the property is missing, None, or empty after stripping.
    """
    try:
        value = getattr(filing_obj, property_name, None)
        if value is None:
            return None
        text = str(value).strip()
        return text if text else None
    except Exception as e:
        logger.debug(f"Could not read property '{property_name}': {e}")
        return None


def _build_section_id(ticker: str, fiscal_year: int, item_number: str) -> str:
    """
    Construct a deterministic section ID.
    Format: {TICKER}_{FISCAL_YEAR}_item_{ITEM_NUMBER}
    Example: AAPL_2024_item_7
    """
    clean_item = item_number.replace(" ", "_").lower()
    return f"{ticker}_{fiscal_year}_item_{clean_item}"


def parse_filing(raw_filing: RawFiling) -> List[ParsedSection]:
    """
    Parse a RawFiling into a list of ParsedSection objects.

    This is the main entry point called by the pipeline.
    It processes all standard 10-K sections plus Item 8 financial statements.

    Args:
        raw_filing: Output of edgar_fetcher.fetch_filing()

    Returns:
        List[ParsedSection] with one entry per non-empty section found.
        Sections that are missing or empty in the filing are silently skipped.

    Example:
        from financevault.ingestion.edgar_fetcher import fetch_filing
        from financevault.ingestion.document_parser import parse_filing

        raw  = fetch_filing("AAPL", fiscal_year=2024)
        sections = parse_filing(raw)

        for s in sections:
            print(s.item_number, s.section_type, s.signals.token_count)
    """
    filing_obj : Any            = raw_filing.filing_object
    metadata   : FilingMetadata = raw_filing.metadata
    ticker     : str            = metadata.ticker
    fiscal_year: int            = metadata.fiscal_year

    parsed_sections: List[ParsedSection] = []

    logger.info(f"[{ticker}] Parsing {len(_SECTION_MAP)} narrative sections + Item 8...")

    # ------------------------------------------------------------------
    # Pass 1: narrative and mixed sections from _SECTION_MAP
    # ------------------------------------------------------------------
    for section_def in _SECTION_MAP:
        prop      = section_def["property"]
        item_num  = section_def["item_number"]
        item_title= section_def["item_title"]
        def_type  = section_def["default_type"]

        raw_text  = _safe_get_text(filing_obj, prop)

        if not raw_text:
            logger.debug(f"[{ticker}] Item {item_num} ({item_title}): empty or missing, skipping.")
            continue

        # Compute signals
        signals = _build_signals(raw_text)

        # Override section type based on signals
        section_type = _classify_section_type(
            default_type      = def_type,
            numerical_density = signals.numerical_density,
            table_density     = signals.table_density,
        )

        # For MIXED sections, extract any embedded tables as markdown strings
        embedded_tables: List[str] = []
        if section_type in (SectionType.MIXED, SectionType.FINANCIAL_TABLE):
            # Find pipe-delimited table blocks within the narrative text
            table_blocks = re.findall(
                r"(\|.+\|(?:\n\|.+\|)+)",
                raw_text,
                re.MULTILINE,
            )
            embedded_tables = [b.strip() for b in table_blocks if b.strip()]

        section_id = _build_section_id(ticker, fiscal_year, item_num)

        parsed_sections.append(
            ParsedSection(
                section_id   = section_id,
                metadata     = metadata,
                item_number  = item_num,
                item_title   = item_title,
                section_type = section_type,
                text         = raw_text,
                tables       = embedded_tables,
                signals      = signals,
                source_url   = raw_filing.raw_html_url,
            )
        )

        logger.info(
            f"[{ticker}] Item {item_num}: {section_type.value}, "
            f"{signals.token_count} tokens, "
            f"num_density={signals.numerical_density:.2f}, "
            f"table_density={signals.table_density:.2f}"
        )

    # ------------------------------------------------------------------
    # Pass 2: Item 8 — financial statements from XBRL
    # ------------------------------------------------------------------
    item8_def   = _ITEM8_DEFINITION
    item8_text, item8_tables = _extract_financial_tables(filing_obj, ticker)

    if item8_text:
        signals_8    = _build_signals(item8_text)
        section_id_8 = _build_section_id(ticker, fiscal_year, item8_def["item_number"])

        # Item 8 is always FINANCIAL_TABLE regardless of detected signals
        parsed_sections.append(
            ParsedSection(
                section_id   = section_id_8,
                metadata     = metadata,
                item_number  = item8_def["item_number"],
                item_title   = item8_def["item_title"],
                section_type = SectionType.FINANCIAL_TABLE,
                text         = item8_text,
                tables       = item8_tables,
                signals      = signals_8,
                source_url   = raw_filing.raw_html_url,
            )
        )
        logger.info(
            f"[{ticker}] Item 8: FINANCIAL_TABLE, "
            f"{signals_8.token_count} tokens, "
            f"{len(item8_tables)} tables serialised."
        )
    else:
        logger.warning(f"[{ticker}] Item 8: No financial statement data extracted.")

    logger.info(
        f"[{ticker}] Parsing complete. "
        f"{len(parsed_sections)} sections extracted."
    )

    return parsed_sections


# ---------------------------------------------------------------------------
# Batch parser — used by the pipeline
# ---------------------------------------------------------------------------

def parse_filings(raw_filings: List[RawFiling]) -> List[ParsedSection]:
    """
    Parse a list of RawFilings into ParsedSections.
    Processes each filing sequentially and accumulates all sections.

    Args:
        raw_filings: Output of edgar_fetcher.fetch_filings()

    Returns:
        Flat list of all ParsedSection objects across all companies.

    Example:
        from financevault.ingestion.edgar_fetcher import fetch_filings
        from financevault.ingestion.document_parser import parse_filings

        raw_list = fetch_filings(["AAPL", "MSFT", "JPM"], fiscal_year=2024)
        all_sections = parse_filings(raw_list)
        print(f"Total sections: {len(all_sections)}")
    """
    all_sections: List[ParsedSection] = []

    for i, raw in enumerate(raw_filings):
        ticker = raw.metadata.ticker
        logger.info(f"Parsing [{i+1}/{len(raw_filings)}]: {ticker}")
        try:
            sections = parse_filing(raw)
            all_sections.extend(sections)
        except Exception as e:
            logger.error(f"[{ticker}] parse_filing raised unexpectedly: {type(e).__name__}: {e}")
            # Never crash the whole batch on one bad filing

    logger.info(
        f"Batch parsing complete. "
        f"{len(all_sections)} total sections from {len(raw_filings)} filings."
    )

    return all_sections