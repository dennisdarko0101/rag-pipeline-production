"""RAG Pipeline — Streamlit demo dashboard.

Communicates with the FastAPI backend via httpx (see ``ui/api_client.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from ui import api_client
from ui.components import metric_card, pipeline_timeline, source_card, status_indicator
from ui.config import COLORS, PAGE_ICON, PAGE_LAYOUT, PAGE_TITLE


def _render_response_details(sources: list, metadata: dict) -> None:
    """Show retrieved sources and the pipeline panel.

    With the wide layout there is room to place them side by side, which keeps
    a typical answer from stacking tall. Falls back to a single full-width panel
    when only one of the two is present.
    """

    def _sources_panel() -> None:
        with st.expander(f"Sources ({len(sources)} chunks)", expanded=False):
            for src in sources:
                source_card(
                    source_name=src["source_name"],
                    chunk_text=src["chunk_text"],
                    chunk_index=src["chunk_index"],
                    relevance_score=src["relevance_score"],
                )

    def _pipeline_panel() -> None:
        with st.expander("Pipeline Metrics", expanded=False):
            pipeline_timeline(metadata)

    if sources and metadata:
        col_src, col_pipe = st.columns([3, 2])
        with col_src:
            _sources_panel()
        with col_pipe:
            _pipeline_panel()
    elif sources:
        _sources_panel()
    elif metadata:
        _pipeline_panel()


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=PAGE_LAYOUT)

# Inject dark-theme CSS overrides
st.markdown(
    f"""
    <style>
        .stApp {{ background-color: {COLORS["bg_surface"]}; }}

        /* Cleaner chrome: hide Streamlit's menu/footer, keep a roomy layout */
        #MainMenu, footer {{ visibility: hidden; }}
        header[data-testid="stHeader"] {{ background: transparent; }}
        /* Use the full wide-layout width (no narrow max-width cap) */
        .block-container {{ max-width: 100%; padding-top: 1rem; padding-bottom: 1.5rem; }}

        /* Compact vertical rhythm so the key content fits a normal viewport */
        [data-testid="stVerticalBlock"] {{ gap: 0.55rem; }}
        h1, h2, h3, h4 {{
            margin-top: 0.1rem !important;
            margin-bottom: 0.3rem !important;
            line-height: 1.25 !important;
        }}
        h3 {{ font-size: 1.15rem !important; }}
        h4 {{ font-size: 1.0rem !important; }}
        [data-testid="stCaptionContainer"] {{ margin-top: -0.15rem; }}
        hr {{ margin: 0.6rem 0 !important; }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
        .stTabs [data-baseweb="tab"] {{
            padding: 6px 16px;
            border-radius: 8px 8px 0 0;
            font-weight: 600;
        }}

        /* Chat messages read as tidy cards */
        .stChatMessage {{
            background: {COLORS["bg_card"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 12px;
            padding: 0.6rem 0.95rem;
            margin-bottom: 0.4rem;
            line-height: 1.55;
        }}

        /* Buttons */
        .stButton > button {{ border-radius: 8px; font-weight: 600; }}

        /* Readability ---------------------------------------------------- */

        /* Default = dark main panel: near-white headings and body text,
           fully opaque. Scoped to headings + markdown paragraphs/lists so it
           never overrides the source/metric cards (those use inline styles on
           divs, which have no <p>). */
        h1, h2, h3, h4,
        .stMarkdown p, .stMarkdown li {{
            color: {COLORS["text"]} !important;
            opacity: 1 !important;
        }}
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p {{
            color: {COLORS["text_secondary"]} !important;
        }}

        /* Expander labels ("Sources (N chunks)", "Pipeline Metrics") */
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span,
        .streamlit-expanderHeader,
        .streamlit-expanderHeader p {{
            color: {COLORS["text"]} !important;
            opacity: 1 !important;
        }}

        /* Light sidebar: dark, legible text. These selectors are more specific
           than the defaults above, so they win for everything in the sidebar. */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
            color: {COLORS["sidebar_text"]} !important;
        }}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
        [data-testid="stSidebar"] small,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {{
            color: {COLORS["sidebar_text_secondary"]} !important;
        }}

        /* Hide the Streamlit "Deploy" button in the top-right */
        [data-testid="stAppDeployButton"],
        [data-testid="stDeployButton"],
        .stDeployButton {{ display: none !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history: list[dict] = []
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
            <span style="font-size:1.6rem;">{PAGE_ICON}</span>
            <span style="font-size:1.3rem; font-weight:700; color:{COLORS["sidebar_text"]};">
                RAG Pipeline
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Configuration panel ----
    st.markdown("#### Configuration")

    provider = st.selectbox(
        "LLM Provider",
        options=["fallback", "claude", "openai"],
        index=0,
        help="fallback = try Claude first, fall back to GPT-4o",
    )

    k = st.slider("Documents to retrieve (k)", min_value=1, max_value=20, value=5)

    rerank = st.toggle("Enable reranking", value=True)

    rerank_top_k = st.slider(
        "Rerank top-k",
        min_value=1,
        max_value=10,
        value=3,
        disabled=not rerank,
    )

    st.divider()

    # ---- System status ----
    st.markdown("#### System Status")

    health = api_client.check_health()
    api_ok = health.get("status") in ("healthy", "degraded")

    status_indicator("API Server", api_ok)

    if api_ok:
        components = health.get("components", {})
        vs = components.get("vectorstore", {})
        vs_ok = vs.get("status") == "healthy"
        status_indicator("Vector Store", vs_ok, vs.get("details", ""))
    else:
        status_indicator("Vector Store", False, health.get("detail", ""))

    st.divider()

    # ---- Document ingestion ----
    st.markdown("#### Ingest Documents")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "md", "txt"],
        help="PDF, Markdown, or plain text",
    )

    if uploaded_file is not None and st.button("Ingest uploaded file", use_container_width=True):
        with st.spinner("Ingesting..."):
            result = api_client.ingest_file(uploaded_file.getvalue(), uploaded_file.name)
        if result.get("error"):
            st.error(result["detail"])
        else:
            st.success(
                f"Processed **{result['documents_processed']}** doc(s), "
                f"**{result['chunks_created']}** chunks created."
            )

    url_input = st.text_input("Or paste a URL", placeholder="https://example.com/article")
    if url_input and st.button("Ingest URL", use_container_width=True):
        with st.spinner("Fetching & ingesting..."):
            result = api_client.ingest_url(url_input)
        if result.get("error"):
            st.error(result["detail"])
        else:
            st.success(
                f"Processed **{result['documents_processed']}** doc(s), "
                f"**{result['chunks_created']}** chunks created."
            )

# ---------------------------------------------------------------------------
# Main area — header + tabs
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div style="margin-bottom:2px;">
        <h1 style="margin:0; font-size:1.4rem; font-weight:700; color:{COLORS["text"]};">
            RAG Pipeline
        </h1>
        <p style="margin:3px 0 0 0; color:{COLORS["text_secondary"]}; font-size:0.9rem;">
            Ask a question and get an answer grounded in your documents, with the exact
            sources it drew from.
        </p>
    </div>
    <hr style="border:none; border-top:1px solid {COLORS["border"]}; margin:8px 0 2px 0;" />
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_eval = st.tabs(["\U0001f4ac  Chat", "\U0001f4ca  Evaluation"])

# ===== CHAT TAB =====
with tab_chat:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show expandable details for assistant messages
            if msg["role"] == "assistant" and "sources" in msg:
                _render_response_details(msg["sources"], msg.get("metadata", {}))

    # Chat input
    if user_input := st.chat_input("Ask a question about your documents..."):
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call the API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = api_client.query(
                    question=user_input,
                    k=k,
                    rerank=rerank,
                    rerank_top_k=rerank_top_k,
                    provider=provider,
                )

            if result.get("error"):
                answer = f"**Error:** {result['detail']}"
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            else:
                answer = result["answer"]
                sources = result.get("sources", [])
                metadata = result.get("metadata", {})

                st.markdown(answer)

                _render_response_details(sources, metadata)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "metadata": metadata,
                    }
                )

# ===== EVALUATION TAB =====
with tab_eval:
    st.markdown("### RAG Evaluation")
    st.caption(
        "Run the golden evaluation dataset against the pipeline and view per-question scores."
    )

    col_btn, col_info = st.columns([1, 3])

    with col_btn:
        run_eval = st.button("Run Evaluation", use_container_width=True)

    # Load golden dataset for the evaluation
    golden_path = Path("tests/eval/eval_dataset.json")

    if run_eval:
        if not golden_path.exists():
            st.error("Golden dataset not found at tests/eval/eval_dataset.json")
        else:
            data = json.loads(golden_path.read_text(encoding="utf-8"))
            qa_pairs = [
                {"question": p["question"], "ground_truth": p["ground_truth"]}
                for p in data["pairs"]
            ]

            with st.spinner(
                f"Evaluating {len(qa_pairs)} questions — each runs the full pipeline plus "
                "LLM-judge scoring, so this can take several minutes. Please keep this tab open."
            ):
                result = api_client.evaluate(
                    qa_pairs=qa_pairs,
                    k=k,
                    rerank=rerank,
                    provider=provider,
                )

            if result.get("error"):
                st.error(f"Evaluation failed: {result['detail']}")
            else:
                st.session_state.eval_results = result

    # Display evaluation results
    eval_data = st.session_state.eval_results
    if eval_data and not eval_data.get("error"):
        eval_result = eval_data.get("result", {})

        # Aggregate metric cards
        st.markdown("#### Aggregate Metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Faithfulness", eval_result.get("avg_faithfulness"))
        with c2:
            metric_card("Answer Relevancy", eval_result.get("avg_answer_relevancy"))
        with c3:
            metric_card("Context Precision", eval_result.get("avg_context_precision"))
        with c4:
            metric_card(
                "Avg Latency",
                None,
                value_text=f"{eval_result.get('avg_latency_ms', 0):.0f} ms",
            )

        # Per-question results table
        st.markdown("#### Per-Question Results")
        results_list = eval_result.get("results", [])
        if results_list:
            table_rows = []
            for i, r in enumerate(results_list, 1):
                table_rows.append(
                    {
                        "#": i,
                        "Question": r["question"][:60] + ("..." if len(r["question"]) > 60 else ""),
                        "Faithfulness": f"{r.get('faithfulness', '-')}"
                        if r.get("faithfulness") is not None
                        else "-",
                        "Relevancy": f"{r.get('answer_relevancy', '-')}"
                        if r.get("answer_relevancy") is not None
                        else "-",
                        "Latency (ms)": f"{r['latency_ms']:.0f}",
                        "Sources": r["num_sources"],
                    }
                )
            # Fixed-height scroll box (~6 rows visible) so the full table does
            # not push the rest of the page down.
            st.dataframe(table_rows, use_container_width=True, hide_index=True, height=245)
        else:
            st.info("No per-question results available.")

        # Previous run comparison placeholder
        st.markdown("#### Run Comparison")
        st.caption(
            "Save evaluation reports with `scripts/run_eval.sh` and compare "
            "runs programmatically via `compare_reports()`."
        )
    elif eval_data is None:
        st.info(
            "Click **Run Evaluation** to evaluate the pipeline against "
            "the golden dataset (18 Q&A pairs)."
        )
