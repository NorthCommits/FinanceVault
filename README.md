# ⬡ FinanceVault

> **Financial Intelligence RAG** — A production-grade Retrieval-Augmented Generation system for SEC 10-K filings, built with adaptive chunking, hybrid retrieval, cross-encoder reranking, and GPT-4o generation.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat&logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Dense%20Search-0075A8?style=flat)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat)

---

## What is FinanceVault?

FinanceVault is a research-grade RAG system that lets you ask natural language questions about SEC 10-K annual filings from 10 major public companies. It goes beyond standard document QA by implementing a full production pipeline with state-of-the-art retrieval and generation techniques.

**Ask questions like:**
- *"What was Apple's gross margin in FY2024?"*
- *"Compare Microsoft and Google revenue growth"*
- *"What are Goldman Sachs's key liquidity risks?"*
- *"How does JPMorgan's capital position compare to industry peers?"*

Every answer comes with source citations, confidence scores, and a full audit trail traceable to the original SEC filing.

---

## Architecture

```
SEC EDGAR (free API)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                   INGESTION LAYER                   │
│                                                     │
│  edgar_fetcher.py   →  document_parser.py           │
│  (fetch 10-K HTML)     (extract 9 sections/filing)  │
│                                                     │
│              ADAPTIVE CHUNKING                      │
│  ┌──────────────────────────────────────────────┐  │
│  │  For each section, run 4 strategies:          │  │
│  │    recursive_600 · recursive_1100             │  │
│  │    table_aware   · llm_regex                  │  │
│  │                                               │  │
│  │  Score each on 8 metrics:                     │  │
│  │    SC · ICC · DCC · BI · RC (paper)           │  │
│  │    NDS · TBI · SPS (FinanceVault custom)      │  │
│  │                                               │  │
│  │  Select winner per section → List[Chunk]      │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                  RETRIEVAL LAYER                    │
│                                                     │
│  text-embedding-3-small  →  FAISS IndexFlatIP       │
│  (OpenAI embeddings)        (dense vector search)   │
│                                                     │
│  BM25Okapi                                          │
│  (financial-aware sparse retrieval)                 │
│                                                     │
│  Reciprocal Rank Fusion (RRF, k=60)                 │
│  BM25 + FAISS → top-20 candidates                  │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                  RERANKING LAYER                    │
│                                                     │
│  BAAI/bge-reranker-large                            │
│  Cross-encoder: (query, chunk) joint scoring        │
│  Sigmoid normalisation → [0, 1] scores              │
│  top-20 → top-5 for generation                      │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                 GENERATION LAYER                    │
│                                                     │
│  Query classification: factual / comparative /      │
│  analytical → specialised system prompt             │
│                                                     │
│  Context assembly: labelled source blocks           │
│  GPT-4o (temp=0.1) → cited answer                  │
│                                                     │
│  Verification: citation validation + confidence     │
│  scoring + audit trail                              │
└─────────────────────────────────────────────────────┘
        │
        ▼
    VerifiedResponse
    (answer · sources · confidence · audit trail)
```

---

## Key Technical Contributions

### 1. Adaptive Chunking (Per-Section Strategy Selection)
Inspired by [ekimetrics/adaptive-chunking](https://github.com/ekimetrics/adaptive-chunking), FinanceVault extends the paper's approach for the financial domain. Instead of applying one chunking strategy to an entire document, we run a competition per section:

- **4 candidate strategies**: `recursive_600`, `recursive_1100`, `table_aware`, `llm_regex`
- **8 scoring metrics**: 5 from the original paper + 3 financial-specific additions
  - `NDS` — Numerical Density Score: numbers stay near their context
  - `TBI` — Table Boundary Integrity: financial table rows never split
  - `SPS` — Section Purity Score: no bleed across Item boundaries
- **Section-type-aware weights**: financial tables weight TBI×0.35, narrative sections weight ICC×0.25

### 2. Hybrid Retrieval with RRF
- **BM25** for exact financial term matching (`EBITDA`, `$391B`, ticker symbols)
- **FAISS** for semantic similarity (`revenue declined` ≈ `sales decreased`)
- **Reciprocal Rank Fusion** combines both without score calibration
- **Metadata filtering** before vector search (company, year, section, item)

### 3. Financial-Domain Context Engineering
- Three specialised system prompts (factual / comparative / analytical)
- Structured source blocks with company, filing year, and item provenance
- Table chunks labelled `[TABLE]` so GPT treats them as structured data
- Temperature=0.1 for near-deterministic financial answers

### 4. Verified Responses with Audit Trail
- Every `[Source N]` citation validated against actual retrieved chunks
- Invalid citations stripped before returning to user
- Confidence scored on three independent signals: retrieval strength, corroboration, hedging language
- Full audit trail: query ID, timestamp, model, retrieval scores per chunk, chunking strategies used

---

## Corpus

| Company | Ticker | Sector | Sections | Tokens |
|---|---|---|---|---|
| Apple Inc. | AAPL | Technology | 4 | 27,723 |
| Microsoft Corp | MSFT | Technology | 4 | 46,106 |
| Alphabet Inc. | GOOGL | Technology | 4 | 41,889 |
| JPMorgan Chase | JPM | Finance | 4 | 47,793 |
| Goldman Sachs | GS | Finance | 4 | 137,587 |
| BlackRock Inc. | BLK | Finance | 4 | 94,525 |
| ExxonMobil Corp | XOM | Energy | 4 | 17,300 |
| Chevron Corp | CVX | Energy | 4 | 34,083 |
| Walmart Inc. | WMT | Retail | 4 | 50,473 |
| Amazon.com Inc. | AMZN | Retail | 4 | 32,246 |

**Total: 40 sections · 982 chunks · 529,725 tokens · FY2024**

All data sourced from SEC EDGAR (free public API, no key required).

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI text-embedding-3-small (1536d) |
| Dense index | FAISS IndexFlatIP |
| Sparse index | rank-bm25 (BM25Okapi) |
| Reranker | BAAI/bge-reranker-large |
| Scoring embedder | sentence-transformers/all-MiniLM-L6-v2 |
| Data source | SEC EDGAR via edgartools |
| Framework | Python 3.10+ |
| UI | Streamlit |
| Data models | Pydantic v2 |

---

## Project Structure

```
FinanceVault/
├── financevault/
│   ├── ingestion/
│   │   ├── models.py              # All Pydantic models
│   │   ├── edgar_fetcher.py       # SEC EDGAR → RawFiling
│   │   ├── document_parser.py     # RawFiling → List[ParsedSection]
│   │   └── chunking/
│   │       ├── strategies.py      # 4 chunking strategies
│   │       ├── scorer.py          # 8 quality metrics
│   │       └── selector.py        # Per-section competition orchestrator
│   ├── retrieval/
│   │   ├── embedder.py            # OpenAI embeddings + batching
│   │   ├── faiss_store.py         # FAISS index with metadata filtering
│   │   ├── bm25_store.py          # BM25 sparse index
│   │   └── hybrid_retriever.py    # RRF fusion → List[RetrievalResult]
│   ├── reranking/
│   │   └── cross_encoder.py       # BGE reranker, sigmoid normalised
│   ├── generation/
│   │   ├── generator.py           # Query classification + GPT-4o
│   │   └── verifier.py            # Citations + confidence + audit trail
│   └── pipeline.py                # End-to-end orchestrator
├── app/
│   └── main.py                    # Streamlit UI
├── data/
│   ├── raw/                       # Filing metadata (10 JSON files)
│   ├── processed/                 # Parsed sections (10 + combined)
│   └── indexes/                   # FAISS + BM25 + chunk store
├── fetch_data.py                  # One-time: fetch from SEC EDGAR
├── run_build.py                   # One-time: chunk + embed + index
└── requirements.txt
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- OpenAI API key

### Installation

```bash
git clone https://github.com/yourusername/FinanceVault.git
cd FinanceVault

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Add your OpenAI API key to .env
echo "OPENAI_API_KEY=sk-..." >> .env
echo "KMP_DUPLICATE_LIB_OK=TRUE" >> .env
```

### Data Collection (one-time, ~10-15 min)

```bash
mkdir -p data/raw data/processed data/indexes
python fetch_data.py
```

Fetches FY2024 10-K filings for all 10 companies from SEC EDGAR (free, no API key needed).

### Build Indexes (one-time, ~5-10 min)

```bash
python run_build.py
```

Runs adaptive chunking, embeds all chunks via OpenAI, builds FAISS and BM25 indexes.
OpenAI embedding cost for the full corpus: ~$0.01.

### Launch the App

```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser.

---

## Example Queries

**Factual**
```
What was Apple's total net sales in FY2024?
→ Filter: Apple (AAPL)
```

**Comparative**
```
Compare gross margins of Apple and Microsoft in FY2024
→ Filter: All Companies
```

**Analytical**
```
What are the key liquidity risks for Goldman Sachs?
→ Filter: Goldman Sachs (GS), Item 1A — Risk Factors
```

---

## Design Decisions

**Why adaptive chunking over fixed-size chunking?**
Financial documents have radically different section structures. Risk factors (Item 1A) consist of dense self-contained paragraphs that chunk well at 600 tokens. Financial statements (Item 8) contain table rows that must never be split. MD&A sections mix prose and tables. A single strategy applied uniformly destroys structure in at least one of these cases.

**Why hybrid retrieval over dense-only?**
BM25 handles exact financial terms (`EBITDA`, `$391,035`, `basis points`) that semantic embeddings collapse. FAISS handles paraphrase and semantic equivalence (`revenue declined` = `sales decreased`). Neither alone achieves the recall of both combined.

**Why RRF over score fusion?**
BM25 scores (unbounded) and cosine similarities ([0,1]) cannot be summed without calibration. RRF uses only rank positions, making it robust to score scale differences and consistently outperforming linear combination in IR benchmarks.

**Why a separate reranker after hybrid retrieval?**
Bi-encoders (FAISS) encode query and chunk independently. Cross-encoders run full attention across both jointly, which is far more accurate but 100x slower. Running the cross-encoder on 20 candidates (not 982) gives accuracy without latency cost.

---

## Roadmap

- [ ] Evaluation suite with precision@k, recall@k, MRR, NDCG
- [ ] LLM_REGEX chunking strategy enabled (improves complex section structure)
- [ ] Multi-year corpus (FY2022, FY2023, FY2024) for trend analysis
- [ ] Multimodal pipeline: extract and embed financial charts from PDF exhibits
- [ ] Fine-tuned reranker on financial (query, passage) pairs
- [ ] Query history and session management in the UI
- [ ] Export answers to PDF with full citation appendix

---

## Research References

- Cormack, Clarke & Buettcher (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods.* SIGIR.
- Ekimetrics (2024). *Adaptive Chunking for RAG.* [github.com/ekimetrics/adaptive-chunking](https://github.com/ekimetrics/adaptive-chunking)
- Xiao et al. (2023). *C-Pack: Packed Resources For General Chinese Embeddings.* (BGE reranker)
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.

---

## Author

**Swapnil Bhattacharya**
AI/GenAI Engineer · MSc Data Science, University of Birmingham

[GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourprofile) · [PyPI](https://pypi.org/user/yourprofile)

---

*Built with real SEC filings, real retrieval engineering, and no shortcuts.*