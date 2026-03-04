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
) -> None:
    """Render a colour-coded metric card.

    Green >= 0.8, yellow >= 0.6, red < 0.6.
    """
    if score is None:
        colour = COLORS["muted"]
        display = "N/A"
        band = ""
    else:
        colour = score_color(score)
        display = f"{score:.2f}"
        band = score_label(score)

    st.markdown(
        f"""
        <div style="
            background: {COLORS["bg_card"]};
            border: 1px solid {COLORS["border"]};
            border-left: 4px solid {colour};
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 8px;
        ">
            <div style="color:{COLORS["text_secondary"]}; font-size:0.8rem;
                         text-transform:uppercase; letter-spacing:0.05em;">
                {label}
            </div>
            <div style="font-size:1.8rem; font-weight:700; color:{colour};
                        margin:4px 0;">
                {display}
            </div>
            <div style="color:{COLORS["text_secondary"]}; font-size:0.75rem;">
                {band}{(" — " + description) if description else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Source card
# ---------------------------------------------------------------------------


def source_card(
    source_name: str,
    chunk_text: str,
    chunk_index: int,
    relevance_score: float,
) -> None:
    """Render a retrieved source with a relevance score bar."""
    bar_width = max(5, int(relevance_score * 100))
    bar_colour = score_color(relevance_score)

    st.markdown(
        f"""
        <div style="
            background: {COLORS["bg_card"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 10px;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center;
                        margin-bottom:8px;">
                <span style="font-weight:600; color:{COLORS["text"]}; font-size:0.9rem;">
                    {source_name}
                    <span style="color:{COLORS["text_secondary"]}; font-weight:400;">
                        &nbsp;chunk {chunk_index}
                    </span>
                </span>
                <span style="color:{bar_colour}; font-weight:600; font-size:0.85rem;">
                    {relevance_score:.2f}
                </span>
            </div>
            <div style="background:{COLORS["border"]}; border-radius:4px; height:6px;
                        margin-bottom:10px;">
                <div style="background:{bar_colour}; width:{bar_width}%;
                            border-radius:4px; height:100%;"></div>
            </div>
            <div style="color:{COLORS["text_secondary"]}; font-size:0.82rem;
                        line-height:1.5; white-space:pre-wrap;">
                {chunk_text[:300]}{"..." if len(chunk_text) > 300 else ""}
            </div>
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
            padding: 16px 20px;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center;
                        margin-bottom:12px;">
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
    """Render a green/red status dot with label."""
    colour = COLORS["success"] if is_healthy else COLORS["danger"]
    status_text = "Online" if is_healthy else "Offline"
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <span style="width:10px; height:10px; border-radius:50%;
                         background:{colour}; display:inline-block;"></span>
            <span style="color:{COLORS["text"]}; font-size:0.85rem;">{label}</span>
            <span style="color:{COLORS["text_secondary"]}; font-size:0.75rem;
                         margin-left:auto;">{status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if detail:
        st.caption(detail)
