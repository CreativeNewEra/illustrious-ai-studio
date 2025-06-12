import json
import logging
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)

# Default MCP server endpoints
MCP_SERVER_URLS: Dict[str, str] = {
    "filesystem": "http://localhost:8001",
    "web_fetch": "http://localhost:8002",
    "git": "http://localhost:8003",
    "image_analysis": "http://localhost:8004",
}


def call_tool(server: str, tool: str, **kwargs: Any) -> str:
    """Call an MCP tool and return the result as a string."""
    base_url = MCP_SERVER_URLS.get(server)
    if not base_url:
        return f"Unknown MCP server: {server}"

    url = f"{base_url}/tools/{tool}"
    try:
        response = requests.post(url, json={"arguments": kwargs}, timeout=30)
    except Exception as e:
        logger.error("Request to %s failed: %s", url, e)
        return f"❌ Request failed: {e}"

    if response.status_code != 200:
        logger.error("MCP server error %s: %s", response.status_code, response.text)
        return f"❌ {response.status_code}: {response.text}"

    # Pretty print JSON if possible
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = response.json()
            return json.dumps(data, indent=2)
        except Exception:
            return response.text
    return response.text
