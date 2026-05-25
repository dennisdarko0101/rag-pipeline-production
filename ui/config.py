"""Streamlit UI configuration: page settings, API URL, theme colours."""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Page config (passed to st.set_page_config)
# ---------------------------------------------------------------------------

PAGE_TITLE = "RAG Pipeline"
PAGE_ICON = "\U0001f50d"  # magnifying glass
PAGE_LAYOUT = "wide"

# ---------------------------------------------------------------------------
# API connection
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "120"))
# Evaluation runs the full pipeline plus three LLM-judge calls per question, so
# the whole batch can take several minutes. It needs a much longer read timeout
# than a single query.
EVAL_TIMEOUT = int(os.getenv("EVAL_TIMEOUT", "1800"))

# ---------------------------------------------------------------------------
# Theme colours
# ---------------------------------------------------------------------------

COLORS = {
    "primary": "#6366f1",  # indigo-500
    "primary_light": "#818cf8",  # indigo-400
    "success": "#22c55e",  # green-500
    "warning": "#eab308",  # yellow-500
    "danger": "#ef4444",  # red-500
    "muted": "#94a3b8",  # slate-400
    "bg_card": "#1e293b",  # slate-800
    "bg_surface": "#0f172a",  # slate-900
    "text": "#f1f5f9",  # slate-100 — near-white body text (dark main panel)
    "text_secondary": "#cbd5e1",  # slate-300 — legible secondary text on navy
    "border": "#334155",  # slate-700
    "sidebar_text": "#1e293b",  # slate-800 — dark text for the light sidebar
    "sidebar_text_secondary": "#475569",  # slate-600 — dark secondary for sidebar captions
}


def score_color(score: float) -> str:
    """Return a colour hex based on metric score."""
    if score >= 0.8:
        return COLORS["success"]
    if score >= 0.6:
        return COLORS["warning"]
    return COLORS["danger"]


def score_label(score: float) -> str:
    """Return a human label for a score band."""
    if score >= 0.8:
        return "Good"
    if score >= 0.6:
        return "Fair"
    return "Poor"
