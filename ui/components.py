"""Reusable Streamlit UI components."""

from __future__ import annotations

import streamlit as st

from ui.config import COLORS, score_color, score_label

# ---------------------------------------------------------------------------
# Metric card
# ---------------------------------------------------------------------------


def metric_card(
    label: str,
    score: float | None,
    description: str = "",
    value_text: str | None = None,
) -> None:
    """Render a metric card.

    For a 0-1 quality score, pass ``score`` (coloured green >= 0.8, yellow >= 0.6,
    red < 0.6, with a band label). For a non-score value such as latency, pass
    ``value_text`` and it is shown as the main value instead.

    The markup is kept on flush-left, joined lines so Streamlit's markdown parser
    does not treat indented HTML as a code block (which would leak raw tags).
    """
    if value_text is not None:
        colour = COLORS["primary_light"]
        display = value_text
        band = ""
    elif score is None:
        colour = COLORS["muted"]
        display = "N/A"
        band = ""
    else:
        colour = score_color(score)
        display = f"{score:.2f}"
        band = score_label(score)

    sub = band + ((" — " + description) if description else "")

    html = (
        f'<div style="background:{COLORS["bg_card"]}; border:1px solid {COLORS["border"]};'
        f" border-left:4px solid {colour}; border-radius:8px; padding:11px 14px;"
        f' margin-bottom:0;">'
        f'<div style="color:{COLORS["text_secondary"]}; font-size:0.75rem;'
        f' text-transform:uppercase; letter-spacing:0.05em;">{label}</div>'
        f'<div style="font-size:1.5rem; font-weight:700; color:{colour};'
        f' margin:2px 0;">{display}</div>'
        f'<div style="color:{COLORS["text_secondary"]}; font-size:0.75rem;">{sub}</div>'
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Source card
# ---------------------------------------------------------------------------


def source_card(
    source_name: str,
    chunk_text: str,
    chunk_index: int,
    relevance_score: float,
) -> None:
    """Render a retrieved source: document name, chunk number, score, and snippet.

    The score is the retriever/reranker's raw relevance score. Cross-encoder
    scores are unbounded (and can be negative), so we show the real value in a
    neutral pill rather than forcing it onto a 0-1 bar that would misrepresent it.
    """
    snippet = chunk_text.strip()
    if len(snippet) > 280:
        snippet = snippet[:280].rstrip() + "…"

    st.markdown(
        f"""
        <div style="
            background: {COLORS["bg_card"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 7px;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center;
                        gap:12px; margin-bottom:6px;">
                <span style="color:{COLORS["text"]}; font-weight:600; font-size:0.9rem;
                             overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                    {source_name}
                    <span style="color:{COLORS["text_secondary"]}; font-weight:400;
                                 font-size:0.8rem;">&nbsp;·&nbsp;chunk {chunk_index}</span>
                </span>
                <span style="color:{COLORS["text_secondary"]}; font-size:0.72rem;
                             background:{COLORS["bg_surface"]}; border:1px solid {COLORS["border"]};
                             border-radius:999px; padding:2px 10px; white-space:nowrap;">
                    score {relevance_score:.2f}
                </span>
            </div>
            <div style="color:{COLORS["text_secondary"]}; font-size:0.84rem;
                        line-height:1.55; white-space:pre-wrap;">{snippet}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pipeline timeline
# ---------------------------------------------------------------------------


def pipeline_timeline(metadata: dict) -> None:
    """Visual timeline showing pipeline stage timings."""
    stages: list[tuple[str, str, str | None]] = [
        ("Retrieval", "retrieve_ms", "\U0001f50d"),
        ("Reranking", "rerank_ms", "\u2b06\ufe0f"),
        ("Generation", "generate_ms", "\u2728"),
    ]

    total_ms = metadata.get("latency_ms", 0)

    st.markdown(
        f"""
        <div style="
            background: {COLORS["bg_card"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 8px;
            padding: 12px 16px;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center;
                        margin-bottom:10px;">
                <span style="font-weight:600; color:{COLORS["text"]};">
                    Pipeline Execution
                </span>
                <span style="color:{COLORS["primary_light"]}; font-weight:600;">
                    {total_ms:.0f} ms total
                </span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    for label, key, icon in stages:
        ms = metadata.get(key)
        if ms is not None:
            pct = (ms / total_ms * 100) if total_ms > 0 else 0
            st.markdown(
                f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between;
                                color:{COLORS["text_secondary"]}; font-size:0.82rem;
                                margin-bottom:3px;">
                        <span>{icon} {label}</span>
                        <span>{ms:.0f} ms ({pct:.0f}%)</span>
                    </div>
                    <div style="background:{COLORS["border"]}; border-radius:3px; height:8px;">
                        <div style="background:{COLORS["primary"]}; width:{max(2, pct):.0f}%;
                                    border-radius:3px; height:100%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Extra stats
    tokens = metadata.get("tokens_used", {})
    num_retrieved = metadata.get("num_retrieved", "?")
    num_reranked = metadata.get("num_reranked", 0)
    input_tok = tokens.get("input_tokens", "?")
    output_tok = tokens.get("output_tokens", "?")

    st.markdown(
        f"""
            <div style="border-top: 1px solid {COLORS["border"]}; margin-top:12px;
                        padding-top:10px; display:flex; gap:24px;
                        color:{COLORS["text_secondary"]}; font-size:0.78rem;">
                <span>Chunks retrieved: <b>{num_retrieved}</b></span>
                <span>After reranking: <b>{num_reranked}</b></span>
                <span>Tokens: <b>{input_tok}</b> in / <b>{output_tok}</b> out</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Status indicator
# ---------------------------------------------------------------------------


def status_indicator(label: str, is_healthy: bool, detail: str = "") -> None:
    """Render a status dot with the label and a colour-coded Online/Offline state."""
    colour = COLORS["success"] if is_healthy else COLORS["danger"]
    status_text = "Online" if is_healthy else "Offline"
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
            <span style="width:9px; height:9px; border-radius:50%;
                         background:{colour}; display:inline-block;
                         box-shadow:0 0 0 3px {colour}22;"></span>
            <span style="color:{COLORS["sidebar_text"]}; font-size:0.85rem;">{label}</span>
            <span style="color:{colour}; font-size:0.75rem; font-weight:600;
                         margin-left:auto;">{status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if detail:
        st.caption(detail)
