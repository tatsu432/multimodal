"""Local Ollama HTTP client — no API key required."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaError(RuntimeError):
    pass


def chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout_sec: float = 300.0,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    logger.info("Calling Ollama model=%s at %s", model, base_url)
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Could not reach Ollama at {base_url}. Is it running? ({exc.reason})"
        ) from exc

    message = body.get("message") or {}
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise OllamaError(f"Ollama returned empty content: {body!r}")
    return content.strip()
