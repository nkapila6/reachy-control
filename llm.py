"""LLM streaming client (OpenAI-compatible chat completions).

Port of the LLM streaming from cmd/reachy-agent/main.go, extended with
OpenAI-style tool calling.
"""

import json
import logging
from typing import Iterator

import requests

logger = logging.getLogger(__name__)


def _parse_arguments(raw: str) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def stream_chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 150,
        temperature: float = 0.7,
    ) -> Iterator[dict]:
        """Yield event dicts.

        {"type": "token", "content": str} for content deltas, and
        {"type": "tool_call", "tool_call": {"id", "name", "arguments"}} for
        finished tool calls.
        """
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if tools is not None:
            body["tools"] = tools

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers=headers,
            stream=True,
            timeout=60,
        )
        if resp.status_code != 200:
            logger.error("llm: HTTP %d: %s", resp.status_code, resp.text)
            return

        # Tool call arguments stream as fragments keyed by index; accumulate
        # them here and emit a finished call once finish_reason is tool_calls.
        pending: dict[int, dict] = {}

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason")

            content = delta.get("content")
            if content:
                yield {"type": "token", "content": content}

            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                entry = pending.setdefault(
                    idx, {"id": None, "name": None, "arguments": ""}
                )
                if tc.get("id"):
                    entry["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    entry["name"] = fn["name"]
                if fn.get("arguments"):
                    entry["arguments"] += fn["arguments"]

            if finish_reason == "tool_calls":
                for idx in sorted(pending):
                    yield self._finish_tool_call(pending[idx], idx)
                pending = {}

        # Stream ended without an explicit tool_calls finish_reason.
        for idx in sorted(pending):
            yield self._finish_tool_call(pending[idx], idx)

    @staticmethod
    def _finish_tool_call(entry: dict, idx: int) -> dict:
        return {
            "type": "tool_call",
            "tool_call": {
                "id": entry["id"] or f"call_{idx}",
                "name": entry["name"] or "",
                "arguments": _parse_arguments(entry["arguments"]),
            },
        }
