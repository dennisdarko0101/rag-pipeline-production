"""HTTP client for the FastAPI backend.

All UI ↔ backend communication goes through this module so the Streamlit
app never imports internal Python modules directly.
"""

from __future__ import annotations

from typing import Any

import httpx

from ui.config import API_BASE_URL, API_TIMEOUT


def _url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def check_health() -> dict[str, Any]:
    """GET /health — returns parsed JSON or error dict."""
    try:
        r = httpx.get(_url("/health"), timeout=10)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        return {"status": "error", "detail": f"HTTP {exc.response.status_code}"}
    except httpx.RequestError as exc:
        return {"status": "unreachable", "detail": str(exc)}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def query(
    question: str,
    k: int = 10,
    rerank: bool = True,
    rerank_top_k: int = 5,
    provider: str = "fallback",
) -> dict[str, Any]:
    """POST /api/v1/query — run the RAG pipeline."""
    payload = {
        "question": question,
        "k": k,
        "rerank": rerank,
        "rerank_top_k": rerank_top_k,
        "provider": provider,
    }
    try:
        r = httpx.post(_url("/api/v1/query"), json=payload, timeout=API_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        body = (
            exc.response.json()
            if exc.response.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        return {
            "error": True,
            "status_code": exc.response.status_code,
            "detail": body.get("detail", str(exc)),
        }
    except httpx.RequestError as exc:
        return {"error": True, "detail": f"Connection error: {exc}"}


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def ingest_file(file_bytes: bytes, filename: str) -> dict[str, Any]:
    """POST /api/v1/ingest/upload — upload and ingest a file."""
    try:
        files = {"file": (filename, file_bytes)}
        r = httpx.post(_url("/api/v1/ingest/upload"), files=files, timeout=API_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        body = (
            exc.response.json()
            if exc.response.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        return {"error": True, "detail": body.get("detail", str(exc))}
    except httpx.RequestError as exc:
        return {"error": True, "detail": f"Connection error: {exc}"}


def ingest_url(url: str) -> dict[str, Any]:
    """POST /api/v1/ingest — ingest from a URL."""
    try:
        r = httpx.post(
            _url("/api/v1/ingest"),
            json={"source_path": url, "doc_type": "web"},
            timeout=API_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        body = (
            exc.response.json()
            if exc.response.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        return {"error": True, "detail": body.get("detail", str(exc))}
    except httpx.RequestError as exc:
        return {"error": True, "detail": f"Connection error: {exc}"}


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------


def evaluate(
    qa_pairs: list[dict[str, str]],
    k: int = 10,
    rerank: bool = True,
    provider: str = "fallback",
) -> dict[str, Any]:
    """POST /api/v1/evaluate — run evaluation against Q&A pairs."""
    payload = {
        "qa_pairs": qa_pairs,
        "k": k,
        "rerank": rerank,
        "provider": provider,
    }
    try:
        r = httpx.post(_url("/api/v1/evaluate"), json=payload, timeout=API_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        body = (
            exc.response.json()
            if exc.response.headers.get("content-type", "").startswith("application/json")
            else {}
        )
        return {"error": True, "detail": body.get("detail", str(exc))}
    except httpx.RequestError as exc:
        return {"error": True, "detail": f"Connection error: {exc}"}
