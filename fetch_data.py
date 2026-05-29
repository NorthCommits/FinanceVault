"""
fetch_data.py

Standalone script to fetch 10-K filings from SEC EDGAR,
parse them into sections, and store everything to disk.

Run from the FinanceVault/ root:
    python fetch_data.py

Output:
    data/raw/          — one JSON file per company with raw filing metadata
    data/processed/    — one JSON file per company with all parsed sections
    data/processed/all_sections.json — combined file with all companies

No API keys required. SEC EDGAR is free and public.
Rate limit: we stay well under 10 requests/second.
Expected runtime: 5-15 minutes for 10 companies depending on network speed.
"""

import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup logging so we can see progress clearly
# ---------------------------------------------------------------------------
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt= "%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/fetch_log.txt", mode="w"),
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Add project root to path so we can import financevault
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from financevault.ingestion.edgar_fetcher import fetch_filing, DEFAULT_TICKERS
from financevault.ingestion.document_parser import parse_filing

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FISCAL_YEAR  = 2024
TICKERS      = DEFAULT_TICKERS   # All 10 companies
RAW_DIR      = Path("data/raw")
PROCESSED_DIR= Path("data/processed")

# Ensure directories exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
Path("data/indexes").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def serialise_section(section) -> dict:
    """Convert a ParsedSection to a JSON-serialisable dict."""
    return {
        "section_id"  : section.section_id,
        "item_number" : section.item_number,
        "item_title"  : section.item_title,
        "section_type": section.section_type.value,
        "text"        : section.text,
        "tables"      : section.tables,
        "source_url"  : section.source_url,
        "signals"     : {
            "token_count"        : section.signals.token_count,
            "numerical_density"  : section.signals.numerical_density,
            "table_density"      : section.signals.table_density,
            "avg_sentence_length": section.signals.avg_sentence_length,
            "has_subsections"    : section.signals.has_subsections,
            "table_row_count"    : section.signals.table_row_count,
            "paragraph_count"    : section.signals.paragraph_count,
        },
        "metadata": {
            "company_name": section.metadata.company_name,
            "ticker"      : section.metadata.ticker,
            "cik"         : section.metadata.cik,
            "fiscal_year" : section.metadata.fiscal_year,
            "filing_date" : str(section.metadata.filing_date),
            "accession_no": section.metadata.accession_no,
            "sector"      : section.metadata.sector,
        },
    }


def serialise_raw_metadata(raw_filing) -> dict:
    """Serialise just the metadata from a RawFiling (not the edgartools object)."""
    m = raw_filing.metadata
    return {
        "company_name": m.company_name,
        "ticker"      : m.ticker,
        "cik"         : m.cik,
        "fiscal_year" : m.fiscal_year,
        "filing_date" : str(m.filing_date),
        "accession_no": m.accession_no,
        "sector"      : m.sector,
        "sic_code"    : m.sic_code,
        "raw_html_url": raw_filing.raw_html_url,
    }


# ---------------------------------------------------------------------------
# Main fetch + parse loop
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("FinanceVault Data Fetch")
    logger.info(f"Tickers    : {TICKERS}")
    logger.info(f"Fiscal year: {FISCAL_YEAR}")
    logger.info(f"Output     : {PROCESSED_DIR.resolve()}")
    logger.info("=" * 60)

    all_sections      = []
    successful_tickers= []
    failed_tickers    = []

    for i, ticker in enumerate(TICKERS):
        logger.info(f"\n[{i+1}/{len(TICKERS)}] Processing {ticker}...")

        # ------------------------------------------------------------------
        # Step 1: Fetch raw filing from EDGAR
        # ------------------------------------------------------------------
        try:
            raw_filing = fetch_filing(
                ticker      = ticker,
                fiscal_year = FISCAL_YEAR,
                sleep_seconds = 0.2,
            )
        except Exception as e:
            logger.error(f"[{ticker}] fetch_filing raised: {e}")
            failed_tickers.append(ticker)
            continue

        if raw_filing is None:
            logger.warning(f"[{ticker}] No filing returned. Skipping.")
            failed_tickers.append(ticker)
            continue

        # Save raw metadata to data/raw/
        raw_path = RAW_DIR / f"{ticker}_2024_raw.json"
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(serialise_raw_metadata(raw_filing), f, indent=2)
        logger.info(f"[{ticker}] Raw metadata saved → {raw_path}")

        # ------------------------------------------------------------------
        # Step 2: Parse into sections
        # ------------------------------------------------------------------
        try:
            sections = parse_filing(raw_filing)
        except Exception as e:
            logger.error(f"[{ticker}] parse_filing raised: {e}")
            failed_tickers.append(ticker)
            continue

        if not sections:
            logger.warning(f"[{ticker}] No sections parsed. Skipping.")
            failed_tickers.append(ticker)
            continue

        # Save parsed sections to data/processed/{TICKER}_2024_sections.json
        ticker_sections = [serialise_section(s) for s in sections]
        processed_path  = PROCESSED_DIR / f"{ticker}_2024_sections.json"
        with open(processed_path, "w", encoding="utf-8") as f:
            json.dump(ticker_sections, f, indent=2, ensure_ascii=False)

        logger.info(
            f"[{ticker}] Saved {len(sections)} sections → {processed_path}"
        )

        # Print section summary
        for s in sections:
            logger.info(
                f"  Item {s.item_number:>3} | {s.section_type.value:<16} | "
                f"{s.signals.token_count:>6} tokens | {s.item_title}"
            )

        all_sections.extend(ticker_sections)
        successful_tickers.append(ticker)

        # Polite pause between companies
        if i < len(TICKERS) - 1:
            logger.info(f"[{ticker}] Done. Waiting 2s before next company...")
            time.sleep(2.0)

    # ------------------------------------------------------------------
    # Step 3: Save combined file
    # ------------------------------------------------------------------
    combined_path = PROCESSED_DIR / "all_sections.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_sections, f, indent=2, ensure_ascii=False)

    logger.info(f"\nCombined file saved → {combined_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("FETCH COMPLETE")
    logger.info(f"Successful : {len(successful_tickers)} — {successful_tickers}")
    logger.info(f"Failed     : {len(failed_tickers)} — {failed_tickers}")
    logger.info(f"Total sections: {len(all_sections)}")
    logger.info(
        f"Total tokens  : "
        f"{sum(s['signals']['token_count'] for s in all_sections):,}"
    )
    logger.info(f"Log saved  : data/fetch_log.txt")
    logger.info("=" * 60)

    # Section type breakdown
    type_counts = {}
    for s in all_sections:
        t = s["section_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    logger.info("Section type breakdown:")
    for t, count in sorted(type_counts.items()):
        logger.info(f"  {t:<20}: {count}")


if __name__ == "__main__":
    main()