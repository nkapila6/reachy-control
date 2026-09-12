"""Plain client functions that drive pinchtab (headed Chrome) via its HTTP API.

These tools navigate pages, fill forms, click elements, and inspect the page
state through pinchtab's accessibility snapshot. They are registered via
agent_tools.init_agent_tools when --pinchtab-url is set.
"""

import json
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)


class PinchTabClient:
    def __init__(self, base_url: str, token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        # pinchtab uses Authorization: Bearer <token> on all API calls.
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.last_tab_id = None
        self.instance_id = None
        # Route all tool calls to the visible browser so the user can see them.
        self._maybe_pin_headed_instance()

    def _maybe_pin_headed_instance(self):
        # The headed instance may still be starting when we launch, or the
        # pinchtab server may not be up yet. Poll until it is running or we
        # exhaust our budget, then continue either way (don't crash - the
        # user may have intentionally started only a headless instance).
        attempts = 15
        for attempt in range(attempts):
            try:
                self.refresh_instance()
                if self.instance_id is not None:
                    return
            except requests.RequestException as e:
                logger.warning(
                    "headed instance discovery attempt %d/%d failed, retrying: %s",
                    attempt + 1,
                    attempts,
                    e,
                )
            if attempt < attempts - 1:
                time.sleep(1)

        logger.error(
            "no running headed instance found after %d attempts; "
            "tool calls will be routed to the headless default and will not be visible",
            attempts,
        )

    def refresh_instance(self):
        """Re-query /instances and pin the first running headed instance."""
        resp = self.session.get(f"{self.base_url}/instances")
        resp.raise_for_status()
        instances = resp.json()
        for instance in instances:
            if (
                instance.get("headless") is False
                and instance.get("status") == "running"
            ):
                self.instance_id = instance.get("id")
                logger.info("pinned headed instance %s", self.instance_id)
                return
        logger.warning("no running headed instance found, using default")
        self.instance_id = None

    def _request_headers(self) -> dict:
        # Add X-Instance-Id to every request once we have pinned an instance.
        if self.instance_id:
            return {"X-Instance-Id": self.instance_id}
        return {}

    def navigate(self, url: str, tab_id: str = None) -> dict:
        """Navigate to a URL and remember the returned tabId."""
        body = {"url": url}
        if tab_id:
            body["tabId"] = tab_id
        resp = self.session.post(
            f"{self.base_url}/navigate", json=body, headers=self._request_headers()
        )
        resp.raise_for_status()
        data = resp.json()
        if "tabId" in data:
            self.last_tab_id = data["tabId"]
        return data

    def snapshot(self, tab_id: str = None) -> dict:
        """Get the interactive accessibility tree."""
        params = {"filter": "interactive"}
        if tab_id:
            params["tabId"] = tab_id
        resp = self.session.get(
            f"{self.base_url}/snapshot", params=params, headers=self._request_headers()
        )
        resp.raise_for_status()
        return resp.json()

    def click(self, ref: str, tab_id: str = None) -> dict:
        """Click an element by ref."""
        body = {"kind": "click", "ref": ref}
        if tab_id:
            body["tabId"] = tab_id
        resp = self.session.post(
            f"{self.base_url}/action", json=body, headers=self._request_headers()
        )
        resp.raise_for_status()
        return resp.json()

    def fill(self, ref: str, text: str, tab_id: str = None) -> dict:
        """Fill an input field by ref."""
        body = {"kind": "fill", "ref": ref, "text": text}
        if tab_id:
            body["tabId"] = tab_id
        resp = self.session.post(
            f"{self.base_url}/action", json=body, headers=self._request_headers()
        )
        resp.raise_for_status()
        return resp.json()

    def press_key(self, key: str, tab_id: str = None) -> dict:
        """Press a keyboard key."""
        body = {"kind": "press", "key": key}
        if tab_id:
            body["tabId"] = tab_id
        resp = self.session.post(
            f"{self.base_url}/action", json=body, headers=self._request_headers()
        )
        resp.raise_for_status()
        return resp.json()

    def get_text(self, tab_id: str = None) -> str:
        """Extract page text content."""
        params = {}
        if tab_id:
            params["tabId"] = tab_id
        resp = self.session.get(
            f"{self.base_url}/text", params=params, headers=self._request_headers()
        )
        resp.raise_for_status()
        return resp.text


pinchtab: PinchTabClient | None = None


def init_pinchtab(base_url: str, token: str | None = None):
    """Create the module-level pinchtab client."""
    global pinchtab
    pinchtab = PinchTabClient(base_url, token)


def _active_tab_id(override: str | None = None) -> str | None:
    return override or (pinchtab.last_tab_id if pinchtab else None)


def _wrap_errors(func):
    """Decorator that turns network errors into LLM-friendly error strings."""

    def wrapper(*args, **kwargs):
        try:
            logger.info("tool call: %s(%s)", func.__name__, args[0] if args else {})
            result = func(*args, **kwargs)
            if isinstance(result, str):
                return result
            return json.dumps(result)
        except requests.RequestException as e:
            logger.warning("tool error in %s: %s", func.__name__, e)
            return json.dumps({"error": str(e)})

    return wrapper


@_wrap_errors
def open_form(params: dict) -> dict:
    url = params.get("url")
    if not url:
        return json.dumps({"error": "missing required parameter 'url'"})

    # Reuse the existing tab so we don't open a new one each call.
    result = pinchtab.navigate(url, pinchtab.last_tab_id)
    tab_id = result.get("tabId") or pinchtab.last_tab_id
    snapshot = pinchtab.snapshot(tab_id)
    return {"tabId": tab_id, "snapshot": snapshot}


@_wrap_errors
def fill_field(params: dict) -> dict:
    ref = params.get("ref")
    text = params.get("text")
    tab_id = _active_tab_id(params.get("tabId"))
    if not ref or text is None:
        return json.dumps({"error": "missing required parameters 'ref' and 'text'"})

    pinchtab.fill(ref, text, tab_id)
    return pinchtab.snapshot(tab_id)


@_wrap_errors
def click_element(params: dict) -> dict:
    ref = params.get("ref")
    tab_id = _active_tab_id(params.get("tabId"))
    if not ref:
        return json.dumps({"error": "missing required parameter 'ref'"})

    pinchtab.click(ref, tab_id)
    return pinchtab.snapshot(tab_id)


@_wrap_errors
def press_key(params: dict) -> dict:
    key = params.get("key")
    tab_id = _active_tab_id(params.get("tabId"))
    if not key:
        return {"error": "missing required parameter 'key'"}

    pinchtab.press_key(key, tab_id)
    return pinchtab.snapshot(tab_id)


@_wrap_errors
def get_page_snapshot(params: dict) -> dict:
    tab_id = _active_tab_id(params.get("tabId"))
    return pinchtab.snapshot(tab_id)


@_wrap_errors
def get_page_text(params: dict) -> dict:
    tab_id = _active_tab_id(params.get("tabId"))
    return {"text": pinchtab.get_text(tab_id)}
