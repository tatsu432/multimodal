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


def list_models(base_url: str = DEFAULT_OLLAMA_BASE_URL, timeout_sec: float = 10.0) -> list[str]:
    """Return installed Ollama model names from GET /api/tags."""
    url = base_url.rstrip("/") + "/api/tags"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Could not reach Ollama at {base_url}. Is it running? ({exc.reason})"
        ) from exc

    models = body.get("models") or []
    names: list[str] = []
    for item in models:
        if isinstance(item, dict):
            name = item.get("name") or item.get("model")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def ensure_model(
    model: str,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout_sec: float = 10.0,
) -> None:
    """Raise OllamaError with pull hints if ``model`` is not installed locally."""
    installed = list_models(base_url, timeout_sec=timeout_sec)
    if any(name == model or name.split(":")[0] == model for name in installed):
        return
    hint = (
        f"Model {model!r} is not installed in Ollama.\n"
        "Pull a vision model first (text-only models like qwen3 cannot see images):\n"
        "  ollama pull llava\n"
        "  ollama pull llama3.2-vision\n"
        "  ollama pull qwen2.5vl:7b\n"
        f"Then set VLM_MODEL to the pulled name (e.g. VLM_MODEL=llava)."
    )
    if installed:
        hint += f"\n\nInstalled models: {', '.join(installed)}"
    raise OllamaError(hint)


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
        if exc.code == 404 and "not found" in detail.lower():
            raise OllamaError(
                f"Ollama model {model!r} not found. Pull a vision model, e.g. "
                f"`ollama pull llava`, then set VLM_MODEL=llava in .env.\n{detail}"
            ) from exc
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


def embeddings(
    *,
    model: str,
    input: list[str],
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    timeout_sec: float = 30.0,
) -> list[list[float]]:
    """Return text embeddings via POST /api/embed.

    Returns a list of float vectors, one per input string.
    Raises OllamaError on HTTP errors or unexpected response shape.
    """
    url = base_url.rstrip("/") + "/api/embed"
    payload = json.dumps({"model": model, "input": input}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    logger.info(
        "Calling Ollama embeddings model=%s at %s, n=%d", model, base_url, len(input)
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaError(f"Ollama embed HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Could not reach Ollama at {base_url}. Is it running? ({exc.reason})"
        ) from exc

    vectors = body.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(input):
        raise OllamaError(f"Unexpected Ollama embed response: {body!r}")
    return vectors
