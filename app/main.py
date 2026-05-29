"""
app/main.py

FinanceVault Streamlit application.

Run from FinanceVault/ root:
    streamlit run app/main.py
"""

import os
import sys
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title      = "FinanceVault",
    page_icon       = "⬡",
    layout          = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Bloomberg Terminal × Modern SaaS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Base ─────────────────────────────────── */
:root {
    --bg-primary   : #0a0d14;
    --bg-secondary : #0f1520;
    --bg-card      : #131926;
    --bg-hover     : #1a2235;
    --border       : #1e2d45;
    --border-light : #243350;
    --amber        : #f5a623;
    --amber-dim    : #a06b12;
    --amber-glow   : rgba(245,166,35,0.12);
    --teal         : #00d4aa;
    --teal-dim     : rgba(0,212,170,0.15);
    --red          : #ff4d6a;
    --text-primary : #e8edf5;
    --text-secondary: #7a8fa8;
    --text-muted   : #3d5270;
    --grid         : rgba(30,45,69,0.6);
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif;
}

/* Grid texture overlay */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(var(--grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--grid) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] > div {
    background-color: var(--bg-secondary) !important;
}

/* ── Typography ───────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Instrument Serif', serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em;
}

code, pre, .mono {
    font-family: 'DM Mono', monospace !important;
}

/* ── Inputs ───────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select,
textarea {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stTextInput"] input:focus,
textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px var(--amber-glow) !important;
}

/* ── Buttons ──────────────────────────────────────── */
[data-testid="stButton"] button {
    background-color: var(--amber) !important;
    color: #0a0d14 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}

[data-testid="stButton"] button:hover {
    background-color: #ffc04d !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px var(--amber-glow) !important;
}

/* ── Selectbox ────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-light) !important;
    color: var(--text-primary) !important;
}

/* ── Metrics ──────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] {
    color: var(--amber) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1.6rem !important;
}

/* ── Expander ─────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Divider ──────────────────────────────────────── */
hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Scrollbar ────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber-dim); }

/* ── Custom components ────────────────────────────── */
.fv-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 0.25rem;
}

.fv-logo {
    font-family: 'Instrument Serif', serif;
    font-size: 2rem;
    color: var(--amber);
    letter-spacing: -0.03em;
}

.fv-tagline {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.fv-answer-box {
    background: var(--bg-card);
    border: 1px solid var(--border-light);
    border-left: 3px solid var(--amber);
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
    font-size: 1rem;
    line-height: 1.75;
    color: var(--text-primary);
}

.fv-source-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.875rem 1rem;
    margin: 0.5rem 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
}

.fv-source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.fv-source-label {
    color: var(--amber);
    font-weight: 500;
    letter-spacing: 0.05em;
}

.fv-source-meta {
    color: var(--text-secondary);
    font-size: 0.7rem;
}

.fv-source-preview {
    color: var(--text-secondary);
    font-size: 0.72rem;
    line-height: 1.5;
    border-top: 1px solid var(--border);
    padding-top: 0.5rem;
    margin-top: 0.5rem;
    white-space: pre-wrap;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
}

.fv-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.badge-factual    { background: rgba(0,212,170,0.15); color: #00d4aa; }
.badge-comparative{ background: rgba(245,166,35,0.15); color: #f5a623; }
.badge-analytical { background: rgba(139,92,246,0.15); color: #a78bfa; }
.badge-table      { background: rgba(59,130,246,0.15); color: #60a5fa; }
.badge-narrative  { background: rgba(16,185,129,0.15); color: #34d399; }

.fv-confidence-bar {
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    margin: 0.5rem 0;
    overflow: hidden;
}

.fv-confidence-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--amber), #ffc04d);
    transition: width 0.8s ease;
}

.fv-stat-row {
    display: flex;
    gap: 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
}

.fv-stat {
    display: flex;
    align-items: center;
    gap: 0.3rem;
}

.fv-stat-val {
    color: var(--amber);
    font-weight: 500;
}

.fv-query-chip {
    display: inline-block;
    background: var(--bg-hover);
    border: 1px solid var(--border-light);
    border-radius: 20px;
    padding: 0.3rem 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin: 0.25rem;
    cursor: pointer;
    transition: all 0.15s ease;
}

.fv-query-chip:hover {
    border-color: var(--amber);
    color: var(--amber);
    background: var(--amber-glow);
}

.fv-section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

.stAlert {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}

.stSpinner > div {
    border-top-color: var(--amber) !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline initialisation — cached so it loads once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the FinanceVault pipeline once and cache it for the session."""
    from financevault.pipeline import FinanceVaultPipeline
    pipeline = FinanceVaultPipeline()
    pipeline._ensure_loaded()
    return pipeline


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="fv-header">
        <span class="fv-logo">⬡ FinanceVault</span>
    </div>
    <div class="fv-tagline">Financial Intelligence RAG</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<div class="fv-section-label">Query Filters</div>', unsafe_allow_html=True)

    COMPANIES = {
        "All Companies": None,
        "Apple (AAPL)"         : "AAPL",
        "Microsoft (MSFT)"     : "MSFT",
        "Alphabet (GOOGL)"     : "GOOGL",
        "JPMorgan (JPM)"       : "JPM",
        "Goldman Sachs (GS)"   : "GS",
        "BlackRock (BLK)"      : "BLK",
        "ExxonMobil (XOM)"     : "XOM",
        "Chevron (CVX)"        : "CVX",
        "Walmart (WMT)"        : "WMT",
        "Amazon (AMZN)"        : "AMZN",
    }

    SECTIONS = {
        "All Sections"         : None,
        "Item 1 — Business"    : "1",
        "Item 1A — Risk Factors": "1A",
        "Item 7 — MD&A"        : "7",
        "Item 8 — Financials"  : "8",
    }

    selected_company = st.selectbox("Company", list(COMPANIES.keys()))
    selected_section = st.selectbox("Section", list(SECTIONS.keys()))

    ticker      = COMPANIES[selected_company]
    item_number = SECTIONS[selected_section]

    filters = {}
    if ticker:
        filters["ticker"] = ticker
    if item_number:
        filters["item_number"] = item_number

    st.markdown("---")
    st.markdown('<div class="fv-section-label">Retrieval Settings</div>', unsafe_allow_html=True)

    top_k = st.slider("Final chunks for generation", min_value=3, max_value=10, value=5)

    st.markdown("---")
    st.markdown('<div class="fv-section-label">Corpus Stats</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#7a8fa8; line-height:2;">
        <div>Companies &nbsp;<span style="color:#f5a623">10</span></div>
        <div>Sections &nbsp;&nbsp;<span style="color:#f5a623">40</span></div>
        <div>Chunks &nbsp;&nbsp;&nbsp;<span style="color:#f5a623">982</span></div>
        <div>Tokens &nbsp;&nbsp;&nbsp;<span style="color:#f5a623">529,725</span></div>
        <div>Fiscal Year <span style="color:#f5a623">2024</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#3d5270; line-height:1.8;">
        Adaptive chunking · Hybrid RRF<br>
        Cross-encoder reranking<br>
        GPT-4o generation · Citation audit
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown("""
<div class="fv-header" style="margin-bottom:0.1rem">
    <span class="fv-logo" style="font-size:2.4rem">FinanceVault</span>
    <span class="fv-tagline" style="font-size:0.75rem">SEC 10-K Intelligence · FY2024</span>
</div>
<div style="font-family:'DM Sans',sans-serif; font-size:0.9rem; color:#7a8fa8; margin-bottom:1.5rem;">
    Ask anything about Apple, Microsoft, Google, JPMorgan, Goldman Sachs,
    BlackRock, ExxonMobil, Chevron, Walmart, or Amazon.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
query = st.text_area(
    "Your question",
    placeholder="e.g. What was Apple's gross margin in FY2024? Compare Microsoft and Google revenue growth. What are Goldman Sachs's key liquidity risks?",
    height=90,
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 5])
with col1:
    run_query = st.button("⬡ ANALYSE", use_container_width=True)

# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------
st.markdown('<div style="margin-top:0.5rem">', unsafe_allow_html=True)
example_queries = [
    "What was Apple's total net sales in FY2024?",
    "Compare gross margins of Apple and Microsoft",
    "What are Goldman Sachs's key liquidity risks?",
    "How did Amazon's operating income change in FY2024?",
    "What is ExxonMobil's capital expenditure strategy?",
    "Compare JPMorgan and Goldman Sachs revenue",
]

st.markdown(
    " ".join([f'<span class="fv-query-chip">{q}</span>' for q in example_queries]),
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Run query
# ---------------------------------------------------------------------------
if run_query and query.strip():

    with st.spinner("Loading FinanceVault indexes..."):
        try:
            pipeline = load_pipeline()
        except Exception as e:
            st.error(f"Failed to load pipeline: {e}")
            st.stop()

    start_time = time.time()

    with st.spinner("Retrieving · Reranking · Generating..."):
        try:
            response = pipeline.query(
                question = query.strip(),
                filters  = filters if filters else None,
                top_k    = top_k,
            )
        except Exception as e:
            st.error(f"Query failed: {type(e).__name__}: {e}")
            st.stop()

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # Response header
    # ------------------------------------------------------------------
    badge_class = {
        "factual"    : "badge-factual",
        "comparative": "badge-comparative",
        "analytical" : "badge-analytical",
    }.get(response.query_type, "badge-factual")

    conf_pct  = int(response.confidence * 100)
    conf_color= "#f5a623" if conf_pct >= 70 else "#ff4d6a" if conf_pct < 50 else "#fbbf24"

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.75rem; flex-wrap:wrap;">
        <span class="fv-badge {badge_class}">{response.query_type}</span>
        <span style="font-family:'DM Mono',monospace; font-size:0.72rem; color:{conf_color};">
            ◆ {conf_pct}% confidence
        </span>
        <span style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#3d5270;">
            {response.tokens_used:,} tokens · {elapsed:.1f}s · {len(response.sources)} sources
        </span>
    </div>
    <div class="fv-confidence-bar">
        <div class="fv-confidence-fill" style="width:{conf_pct}%"></div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Answer
    # ------------------------------------------------------------------
    st.markdown('<div class="fv-section-label" style="margin-top:1rem">Answer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="fv-answer-box">{response.answer}</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------
    if response.sources:
        st.markdown('<div class="fv-section-label" style="margin-top:1.5rem">Sources</div>', unsafe_allow_html=True)

        for source in response.sources:
            section_badge = "badge-table" if source.is_table_chunk else "badge-narrative"
            section_label = "TABLE" if source.is_table_chunk else "TEXT"

            ce_score = f"{source.cross_encoder_score:.3f}" if source.cross_encoder_score else "—"

            st.markdown(f"""
            <div class="fv-source-card">
                <div class="fv-source-header">
                    <span class="fv-source-label">[Source {source.source_num}]
                        {source.company_name} · FY{source.fiscal_year} · Item {source.item_number}
                    </span>
                    <span class="fv-badge {section_badge}">{section_label}</span>
                </div>
                <div class="fv-source-meta">
                    {source.item_title} &nbsp;·&nbsp;
                    Strategy: {source.chunking_strategy} &nbsp;·&nbsp;
                    RRF: {source.rrf_score:.4f} &nbsp;·&nbsp;
                    CE: {ce_score} &nbsp;·&nbsp;
                    Chunk score: {source.chunking_score:.3f}
                </div>
                <div class="fv-source-preview">{source.text_preview}</div>
            </div>
            """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------
    with st.expander("Audit Trail", expanded=False):
        audit = response.audit_trail
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Chunks Retrieved", audit.chunks_retrieved)
        col2.metric("Chunks Reranked", audit.chunks_reranked)
        col3.metric("Sources Cited", len(audit.sources_cited))
        col4.metric("Invalid Citations", len(audit.invalid_citations))

        st.markdown("---")
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#7a8fa8; line-height:2;">
            <div>Query ID &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{audit.query_id}</div>
            <div>Timestamp &nbsp;&nbsp;&nbsp;{audit.timestamp_utc}</div>
            <div>Model &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{audit.model_used}</div>
            <div>Query type &nbsp;&nbsp;{audit.query_type}</div>
            <div>Strategies &nbsp;&nbsp;{', '.join(set(audit.chunking_strategies))}</div>
        </div>
        """, unsafe_allow_html=True)

        if audit.retrieval_scores:
            st.markdown("---")
            st.markdown('<div class="fv-section-label">Retrieval Scores</div>', unsafe_allow_html=True)
            for s in audit.retrieval_scores:
                ce = f"{s['cross_encoder_score']:.3f}" if s.get('cross_encoder_score') else "—"
                st.markdown(f"""
                <div style="font-family:'DM Mono',monospace; font-size:0.7rem;
                            color:#7a8fa8; padding:0.3rem 0; border-bottom:1px solid #1e2d45;">
                    <span style="color:#f5a623">[{s['source_num']}]</span>
                    {s['chunk_id']} &nbsp;·&nbsp;
                    RRF <span style="color:#e8edf5">{s['rrf_score']:.4f}</span> &nbsp;·&nbsp;
                    CE <span style="color:#e8edf5">{ce}</span> &nbsp;·&nbsp;
                    BM25 rank <span style="color:#e8edf5">{s.get('bm25_rank', '—')}</span> &nbsp;·&nbsp;
                    FAISS rank <span style="color:#e8edf5">{s.get('faiss_rank', '—')}</span>
                </div>
                """, unsafe_allow_html=True)

elif run_query and not query.strip():
    st.warning("Please enter a question.")

# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------
else:
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        color: #3d5270;
        font-family: 'DM Mono', monospace;
    ">
        <div style="font-size:3rem; margin-bottom:1rem; opacity:0.3;">⬡</div>
        <div style="font-size:0.8rem; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.5rem;">
            Ready to query
        </div>
        <div style="font-size:0.72rem; color:#243350;">
            10 companies · 40 sections · 982 chunks · FY2024
        </div>
    </div>
    """, unsafe_allow_html=True)