"""Tool registry for the local agent loop.

Exposes OpenAI-format tool specs and a dispatcher so main.py stays clean.
Wires the Exa web_search tool and the pinchtab browser tools.
"""

import json
import logging

import context_tools
import pinchtab_tools

logger = logging.getLogger(__name__)

# OpenAI tool definitions. Parameter names must match the underlying functions.
WEB_SEARCH_SPEC = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for live or factual information. Returns titles, "
            "URLs, descriptions, and page content for the top results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "numResults": {
                    "type": "integer",
                    "description": "Number of results to return (default 10)",
                },
            },
            "required": ["query"],
        },
    },
}

PINCHTAB_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "open_form",
            "description": "Navigate the browser to a URL and return the page snapshot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to open"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fill_field",
            "description": "Fill an input field by its accessibility ref.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Element ref from the snapshot",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type into the field",
                    },
                    "tabId": {"type": "string", "description": "Optional tab id"},
                },
                "required": ["ref", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click_element",
            "description": "Click an element by its accessibility ref.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Element ref from the snapshot",
                    },
                    "tabId": {"type": "string", "description": "Optional tab id"},
                },
                "required": ["ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Press a keyboard key in the browser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Key to press"},
                    "tabId": {"type": "string", "description": "Optional tab id"},
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_snapshot",
            "description": "Get the interactive accessibility tree of the current page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tabId": {"type": "string", "description": "Optional tab id"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_text",
            "description": "Extract the text content of the current page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tabId": {"type": "string", "description": "Optional tab id"},
                },
            },
        },
    },
]

# Map tool name -> callable taking a params dict.
_DISPATCH = {
    "web_search": context_tools.web_search,
    "open_form": pinchtab_tools.open_form,
    "fill_field": pinchtab_tools.fill_field,
    "click_element": pinchtab_tools.click_element,
    "press_key": pinchtab_tools.press_key,
    "get_page_snapshot": pinchtab_tools.get_page_snapshot,
    "get_page_text": pinchtab_tools.get_page_text,
}


def init_agent_tools(
    pinchtab_url: str | None = None,
    pinchtab_token: str | None = None,
    exa_key: str | None = None,
) -> list[dict]:
    """Init the tool backends and return the effective tool spec list.

    Tools are included only when their backend is configured: web_search needs
    an Exa key, pinchtab tools need a pinchtab URL.
    """
    specs: list[dict] = []

    if exa_key:
        context_tools.init_context(exa_key)
        specs.append(WEB_SEARCH_SPEC)
    else:
        logger.warning("EXA_API_KEY not set; web_search tool disabled")

    if pinchtab_url:
        pinchtab_tools.init_pinchtab(pinchtab_url, pinchtab_token)
        specs.extend(PINCHTAB_SPECS)
    else:
        logger.warning("pinchtab URL not set; browser tools disabled")

    return specs


def execute_tool(name: str, arguments: dict) -> dict:
    """Run a tool and return a message-ready result dict."""
    func = _DISPATCH.get(name)
    if func is None:
        return {"name": name, "content": json.dumps({"error": f"unknown tool {name}"})}

    try:
        result = func(arguments)
    except Exception as e:
        logger.warning("tool %s failed: %s", name, e)
        return {"name": name, "content": json.dumps({"error": str(e)})}

    # Tool results must be strings for the chat message content.
    if isinstance(result, str):
        content = result
    else:
        content = json.dumps(result)
    return {"name": name, "content": content}
