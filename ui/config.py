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
    "text": "#f1f5f9",  # slate-100
    "text_secondary": "#94a3b8",  # slate-400
    "border": "#334155",  # slate-700
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
