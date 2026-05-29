"""
financevault/ingestion/edgar_fetcher.py

Fetches 10-K filings from SEC EDGAR using edgartools.
Returns a list of RawFiling objects ready for document_parser.py.

Responsibilities:
  - Accept a list of tickers
  - Resolve each ticker to a Company via edgartools
  - Fetch the latest (or a specific year's) 10-K filing
  - Extract basic filing metadata (CIK, accession number, filing date, fiscal year)
  - Wrap everything in our RawFiling Pydantic model
  - Handle rate limits, missing filings, and malformed responses gracefully

What this file does NOT do:
  - Parse section text  (that is document_parser.py)
  - Extract tables      (that is document_parser.py)
  - Chunk anything      (that is chunking/selector.py)

SEC EDGAR rate limit: 10 requests/second.
edgartools handles this internally but we add our own sleep buffer to be safe.
User-Agent header is required by SEC. Set via edgartools' set_identity().
"""

import logging
import time
from datetime import date
from typing import List, Optional

from edgar import Company, set_identity

from .models import FilingMetadata, RawFiling

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SEC compliance: edgartools requires a User-Agent identity string.
# Format: "Name email@domain.com"
# Set once at module load time.
# ---------------------------------------------------------------------------
_IDENTITY = "FinanceVault project financevault@research.com"


def _init_edgar() -> None:
    """Set the SEC-required User-Agent identity for all edgartools requests."""
    set_identity(_IDENTITY)


# ---------------------------------------------------------------------------
# SIC code → sector mapping (partial, covers our 10 target companies)
# Full list: https://www.sec.gov/info/edgar/siccodes.htm
# ---------------------------------------------------------------------------
_SIC_TO_SECTOR: dict[str, str] = {
    "7372": "Technology",
    "7371": "Technology",
    "5045": "Technology",
    "6022": "Finance",
    "6211": "Finance",
    "6282": "Finance",
    "2911": "Energy",
    "1311": "Energy",
    "5311": "Retail",
    "5961": "Retail",
}


def _resolve_sector(sic_code: Optional[str]) -> Optional[str]:
    """Map SIC code to a human-readable sector. Returns None if unknown."""
    if not sic_code:
        return None
    return _SIC_TO_SECTOR.get(sic_code.strip(), "Other")


# ---------------------------------------------------------------------------
# Core fetch function
# ---------------------------------------------------------------------------

def fetch_filing(
    ticker: str,
    fiscal_year: Optional[int] = None,
    sleep_seconds: float = 0.15,
) -> Optional[RawFiling]:
    """
    Fetch one 10-K filing for a given ticker.

    Args:
        ticker:        Exchange ticker, e.g. "AAPL".
        fiscal_year:   If provided, fetch the 10-K whose period covers that
                       fiscal year (e.g. 2024). If None, fetch the latest.
        sleep_seconds: Polite delay between EDGAR requests.
                       Default 0.15s keeps us well under the 10 req/s limit.

    Returns:
        RawFiling if successful, None if the filing could not be fetched.
    """
    _init_edgar()
    ticker = ticker.upper().strip()

    try:
        logger.info(f"[{ticker}] Resolving company via edgartools...")
        company = Company(ticker)

        # Grab filing history for 10-K form type
        filings = company.get_filings(form="10-K")

        if not filings:
            logger.warning(f"[{ticker}] No 10-K filings found on EDGAR.")
            return None

        # Select the target filing
        if fiscal_year is not None:
            # Filter by period_of_report year
            target_filing = None
            for f in filings:
                period = getattr(f, "period_of_report", None)
                if period and str(period).startswith(str(fiscal_year)):
                    target_filing = f
                    break
            if target_filing is None:
                logger.warning(
                    f"[{ticker}] No 10-K found for fiscal year {fiscal_year}. "
                    f"Falling back to latest."
                )
                target_filing = filings.latest()
        else:
            target_filing = filings.latest()

        time.sleep(sleep_seconds)

        # Pull metadata fields from the filing object
        # edgartools exposes these as attributes on the Filing
        cik_raw       = str(getattr(company, "cik", "")).strip()
        accession_no  = str(getattr(target_filing, "accession_number", "")).strip()
        filing_date_raw = getattr(target_filing, "filing_date", None)
        period_raw    = getattr(target_filing, "period_of_report", None)
        sic_code      = str(getattr(company, "sic", "") or "").strip()
        company_name  = str(getattr(company, "name", ticker)).strip()

        # Resolve filing date
        if isinstance(filing_date_raw, date):
            filing_date = filing_date_raw
        elif isinstance(filing_date_raw, str) and filing_date_raw:
            filing_date = date.fromisoformat(filing_date_raw[:10])
        else:
            filing_date = date.today()
            logger.warning(f"[{ticker}] Could not parse filing_date, using today.")

        # Resolve fiscal year from period_of_report
        if fiscal_year is None:
            if period_raw:
                resolved_year = int(str(period_raw)[:4])
            else:
                resolved_year = filing_date.year
        else:
            resolved_year = fiscal_year

        # Build accession number in standard format if missing dashes
        if accession_no and "-" not in accession_no and len(accession_no) == 18:
            accession_no = f"{accession_no[:10]}-{accession_no[10:12]}-{accession_no[12:]}"

        # Raw HTML URL for reference (edgartools constructs this internally)
        raw_html_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={cik_raw}&type=10-K&dateb=&owner=include&count=10"
        )

        metadata = FilingMetadata(
            company_name = company_name,
            ticker       = ticker,
            cik          = cik_raw,
            fiscal_year  = resolved_year,
            filing_date  = filing_date,
            accession_no = accession_no,
            sector       = _resolve_sector(sic_code),
            sic_code     = sic_code or None,
        )

        # Parse the filing into an edgartools TenK object
        # This is the heavy step — edgartools fetches and parses the HTML here
        logger.info(f"[{ticker}] Parsing TenK object (fiscal year {resolved_year})...")
        time.sleep(sleep_seconds)
        filing_object = target_filing.obj()

        if filing_object is None:
            logger.warning(f"[{ticker}] edgartools returned None for .obj() — skipping.")
            return None

        logger.info(f"[{ticker}] Successfully fetched 10-K for fiscal year {resolved_year}.")

        return RawFiling(
            metadata       = metadata,
            filing_object  = filing_object,
            raw_html_url   = raw_html_url,
        )

    except Exception as e:
        logger.error(f"[{ticker}] Failed to fetch 10-K: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Batch fetch — the main entry point used by the pipeline
# ---------------------------------------------------------------------------

def fetch_filings(
    tickers: List[str],
    fiscal_year: Optional[int] = None,
    sleep_seconds: float = 0.15,
) -> List[RawFiling]:
    """
    Fetch 10-K filings for a list of tickers.

    Args:
        tickers:       List of ticker symbols, e.g. ["AAPL", "MSFT", "GOOGL"].
        fiscal_year:   Fetch the 10-K covering this fiscal year for all tickers.
                       If None, fetches the latest available for each.
        sleep_seconds: Polite delay between EDGAR requests per ticker.

    Returns:
        List of RawFiling objects. Tickers that failed are silently skipped
        with an error logged. Never raises — the pipeline handles partial results.

    Example:
        from financevault.ingestion.edgar_fetcher import fetch_filings

        filings = fetch_filings(
            tickers=["AAPL", "MSFT", "JPM", "XOM"],
            fiscal_year=2024,
        )
        print(f"Fetched {len(filings)} filings")
        for f in filings:
            print(f.metadata.ticker, f.metadata.fiscal_year, f.metadata.filing_date)
    """
    _init_edgar()

    results: List[RawFiling] = []
    failed:  List[str]       = []

    for i, ticker in enumerate(tickers):
        logger.info(f"Fetching [{i+1}/{len(tickers)}]: {ticker}")
        filing = fetch_filing(
            ticker       = ticker,
            fiscal_year  = fiscal_year,
            sleep_seconds= sleep_seconds,
        )
        if filing is not None:
            results.append(filing)
        else:
            failed.append(ticker)

        # Extra sleep between companies to avoid EDGAR rate limit bursts
        if i < len(tickers) - 1:
            time.sleep(sleep_seconds * 2)

    logger.info(
        f"Fetch complete. Success: {len(results)}/{len(tickers)}. "
        f"Failed: {failed if failed else 'none'}"
    )

    return results


# ---------------------------------------------------------------------------
# Default ticker list — our 10 target companies across 4 sectors
# ---------------------------------------------------------------------------

DEFAULT_TICKERS: List[str] = [
    # Technology
    "AAPL", "MSFT", "GOOGL",
    # Finance
    "JPM", "GS", "BLK",
    # Energy
    "XOM", "CVX",
    # Retail
    "WMT", "AMZN",
]