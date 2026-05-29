"""
run_build.py

One-time build script for FinanceVault.
Run this once after fetch_data.py to populate data/indexes/.

What it does:
    1. Loads 40 parsed sections from data/processed/all_sections.json
    2. Runs adaptive chunking across all sections
    3. Embeds all chunks using OpenAI text-embedding-3-small
    4. Builds FAISS dense index
    5. Builds BM25 sparse index
    6. Saves everything to data/indexes/

Run from FinanceVault/ root:
    python run_build.py

Options (edit below):
    USE_LLM_CHUNKING = True   — enables LLM_REGEX strategy (costs ~$0.50-$1)
                       False  — uses only recursive + table-aware (free)

After this completes, run the app:
    streamlit run app/main.py
"""

import logging
import sys
import time
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt  = "%H:%M:%S",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/build_log.txt", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — edit these if needed
# ---------------------------------------------------------------------------
USE_LLM_CHUNKING = False   # Set True to enable LLM_REGEX strategy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start = time.time()

    logger.info("=" * 60)
    logger.info("FinanceVault — Build Pipeline")
    logger.info(f"LLM chunking : {'enabled' if USE_LLM_CHUNKING else 'disabled'}")
    logger.info("=" * 60)

    # Ensure indexes dir exists
    Path("data/indexes").mkdir(parents=True, exist_ok=True)

    # Import here so logging is set up first
    from financevault.pipeline import FinanceVaultPipeline

    pipeline = FinanceVaultPipeline(use_llm_chunking=USE_LLM_CHUNKING)

    pipeline.build(force_rechunk= True, force_reembed= True)

    elapsed = time.time() - start
    logger.info(f"\nTotal build time: {elapsed/60:.1f} minutes")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    stats = pipeline.get_stats()
    logger.info("\n--- BUILD STATS ---")
    for key, value in stats.items():
        logger.info(f"  {key:<15}: {value}")

    # ------------------------------------------------------------------
    # Quick sanity check — run one test query
    # ------------------------------------------------------------------
    logger.info("\n--- SANITY CHECK ---")
    logger.info("Running one test query to confirm the pipeline works...\n")

    try:
        response = pipeline.query(
            "What was Apple's total net sales in fiscal 2024?",
            filters={"ticker": "AAPL"},
        )
        logger.info(f"Query type : {response.query_type}")
        logger.info(f"Confidence : {response.confidence:.0%}")
        logger.info(f"Sources    : {len(response.sources)}")
        logger.info(f"Tokens used: {response.tokens_used}")
        logger.info(f"\nAnswer preview:\n{response.answer[:300]}...")
        logger.info("\nSanity check passed.")

    except Exception as e:
        logger.error(f"Sanity check failed: {type(e).__name__}: {e}")
        logger.error("Check your OPENAI_API_KEY in .env and try again.")

    logger.info("\nBuild complete. Run the app with:")
    logger.info("  streamlit run app/main.py")


if __name__ == "__main__":
    main()