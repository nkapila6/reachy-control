"""Client tools that search the web via Exa.

Called by agent_tools.execute_tool during LLM tool-calling rounds. Exposes
web_search, search_and_summarize, fetch_and_summarize, and llm_context. Exa
returns titles, URLs, snippets, and full page text.
"""

from __future__ import annotations

import json
import logging
import os
import re

import requests

logger = logging.getLogger(__name__)

EXA_BASE = "https://api.exa.ai"

# Module-level key, set by init_context.
_api_key: str | None = None


class ExaError(Exception):
    """Raised when an Exa API call fails."""


def init_context(api_key: str | None = None) -> None:
    """Set the Exa API key. Called from main.py at startup."""
    global _api_key
    _api_key = api_key or os.environ.get("EXA_API_KEY")
    if not _api_key:
        logger.warning("EXA_API_KEY not set; web_search tool will return an error")
    else:
        logger.info("Exa web_search ready")


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_api_key}",
        "Content-Type": "application/json",
    }


def _clamp_num_results(n: int | None) -> int:
    if n is None:
        return 10
    try:
        n = int(n)
    except (TypeError, ValueError):
        return 10
    return max(1, min(100, n))


def _wrap_errors(func):
    """Decorator that turns exceptions into LLM-friendly JSON error strings."""

    def wrapper(*args, **kwargs):
        try:
            logger.info("tool call: %s(%s)", func.__name__, args[0] if args else {})
            result = func(*args, **kwargs)
            if isinstance(result, str):
                return result
            return json.dumps(result)
        except Exception as e:
            logger.warning("tool error in %s: %s", func.__name__, e)
            return json.dumps({"error": str(e)})

    return wrapper


@_wrap_errors
def web_search(params: dict) -> dict:
    """Search the web and return a trimmed, JSON-serializable result dict."""
    query = params.get("query")
    if not query:
        return {"error": "missing required parameter 'query'"}

    if not _api_key:
        return {"error": "EXA_API_KEY not set on the robot"}

    num_results = _clamp_num_results(params.get("numResults"))

    payload = {
        "query": query,
        "numResults": num_results,
        "contents": {
            "text": {"maxCharacters": 2000},
        },
    }

    try:
        resp = requests.post(
            f"{EXA_BASE}/search",
            json=payload,
            headers=_headers(),
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return {"error": f"Exa search failed: {e}"}

    trimmed = []
    for r in data.get("results", []):
        text = r.get("text") or ""
        if len(text) > 2000:
            text = text[:2000] + " ... (truncated)"
        trimmed.append(
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "description": r.get("summary") or "",
                "content": text,
            }
        )

    return {
        "query": data.get("query", query),
        "results": trimmed,
    }


def search_and_summarize(query: str, limit: int = 3, max_chars: int = 2000) -> str:
    """Search the web and format the top results for LLM context injection."""
    if not _api_key:
        raise ExaError("EXA_API_KEY not set")

    num_results = max(1, min(100, limit))
    payload = {
        "query": query,
        "numResults": num_results,
        "contents": {"highlights": True},
    }

    try:
        resp = requests.post(
            f"{EXA_BASE}/search",
            json=payload,
            headers=_headers(),
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise ExaError(f"Exa search failed: {e}") from e

    results = data.get("results", [])
    if not results:
        raise ExaError("No search results returned")

    parts = [f'Web search results for: "{query}"']
    for i, r in enumerate(results, start=1):
        highlights = r.get("highlights") or []
        summary = r.get("summary")
        if highlights:
            snippet = " ".join(highlights).strip()
        elif summary:
            snippet = summary.strip()
        else:
            snippet = r.get("title", "")
        parts.append(f"{i}. {r.get('title', '')}\n   {r.get('url', '')}\n   {snippet}")

    full = "\n\n".join(parts)
    if max_chars > 0 and len(full) > max_chars:
        full = full[:max_chars] + "..."
    return full


def fetch_and_summarize(url: str, max_chars: int = 2000) -> str:
    """Fetch a single URL and return a concise, formatted summary."""
    if not _api_key:
        raise ExaError("EXA_API_KEY not set")

    payload = {
        "urls": [url],
        "text": {"maxCharacters": max_chars},
    }

    try:
        resp = requests.post(
            f"{EXA_BASE}/contents",
            json=payload,
            headers=_headers(),
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        raise ExaError(f"Exa contents fetch failed: {e}") from e

    results = data.get("results", [])
    if not results:
        raise ExaError(f"No content returned for {url}")

    result = results[0]
    title = result.get("title", "")
    text = result.get("text", "")
    summary = result.get("summary", "")

    parts = []
    if title:
        parts.append(f"Title: {title}")
    parts.append(f"Source: {url}")
    if summary:
        parts.append(f"Description: {summary}")
    if text:
        parts.append(f"Content:\n{text}")

    full = "\n\n".join(parts)
    if max_chars > 0 and len(full) > max_chars:
        full = full[:max_chars] + "..."
    return full


def _looks_like_url(text: str) -> str | None:
    for token in text.split():
        if token.startswith("http://") or token.startswith("https://"):
            return token
    return None


def _is_question(text: str) -> bool:
    lowered = text.strip().lower()
    prefixes = (
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "tell me",
        "search",
        "find",
        "look up",
    )
    if any(lowered.startswith(p) for p in prefixes):
        return True
    return "?" in lowered


def llm_context(text: str, max_chars: int = 2000) -> str:
    """Build prompt-ready context from a URL or a search question."""
    if not _api_key:
        return ""

    url = _looks_like_url(text)
    if url:
        try:
            return fetch_and_summarize(url, max_chars)
        except ExaError:
            return ""

    if _is_question(text):
        try:
            return search_and_summarize(text, 3, max_chars)
        except ExaError:
            return ""

    return ""


def register_context_tools(client_tools):
    """Register Exa tools with an ElevenLabs ClientTools instance."""
    from elevenlabs.conversational_ai.conversation import ClientTools

    client_tools.register("web_search", web_search)
