"""OpenRouter LLM provider for ARIS.

OpenRouter exposes an OpenAI-compatible API at https://openrouter.ai/api/v1.
Free-tier models (suffixed ':free') require no payment and count against the
free daily quota instead.

Free-tier rate limits on OpenRouter are aggressive (~20 req/min on most :free
models). This provider honours the Retry-After header on 429 responses and
applies a small client-side pacing delay so we stay under the limit.

Default model: meta-llama/llama-3.3-70b-instruct:free
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_MAX_RETRIES = 5

# Client-side pacing: minimum gap between requests across the whole process.
# 3.5s gives us ~17 calls/min — comfortably under the 20/min free-tier ceiling.
_MIN_REQUEST_INTERVAL_SECONDS = 3.5
_last_request_lock = threading.Lock()
_last_request_at = 0.0


def _pace_request() -> None:
    """Block until the global per-process request interval has elapsed."""
    global _last_request_at
    with _last_request_lock:
        wait = (_last_request_at + _MIN_REQUEST_INTERVAL_SECONDS) - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


class OpenRouterProvider:
    """Calls the OpenRouter chat completions endpoint (OpenAI-compatible)."""

    def __init__(self, *, model: str, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "LLM_API_KEY must be set to your OpenRouter API key. "
                "Get one free at https://openrouter.ai/keys"
            )
        self._model = model
        self._api_key = api_key

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        response_format: dict | None = None,
    ) -> str:
        body: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            body["response_format"] = response_format
        payload = json.dumps(body).encode()

        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            _pace_request()
            req = urllib.request.Request(
                f"{_OPENROUTER_BASE}/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/aris-platform",
                    "X-Title": "ARIS Research Intelligence",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw = resp.read().decode()
                body_json = json.loads(raw)
                choice = (body_json.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                return msg.get("content") or ""
            except urllib.error.HTTPError as exc:
                last_exc = exc
                # 429 rate limit / 5xx — retry; 4xx other → fail fast
                if exc.code in (429, 500, 502, 503, 504) and attempt < _MAX_RETRIES:
                    sleep_s = _retry_delay_seconds(exc, attempt)
                    logger.warning(
                        "OpenRouter HTTP %s - retrying in %.1fs (attempt %d/%d)",
                        exc.code, sleep_s, attempt + 1, _MAX_RETRIES + 1,
                    )
                    time.sleep(sleep_s)
                    continue
                raise
            except (TimeoutError, urllib.error.URLError) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    time.sleep(min(30, 2 ** attempt))
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        return ""

    def complete_json(self, prompt: str, *, max_tokens: int = 512) -> dict:
        # Try OpenAI-style json_object response_format first; many OpenRouter
        # models support it and return clean JSON.
        full_prompt = (
            prompt
            + "\n\nRespond with valid JSON only. Do not include markdown code fences "
              "or commentary outside the JSON object."
        )
        try:
            text = self.complete(
                full_prompt,
                max_tokens=max_tokens,
                temperature=0.1,
                response_format={"type": "json_object"},
            )
        except urllib.error.HTTPError as exc:
            # Some models reject response_format → retry without it
            if exc.code in (400, 422):
                text = self.complete(full_prompt, max_tokens=max_tokens, temperature=0.1)
            else:
                raise

        return _parse_json_with_repair(text)


def _retry_delay_seconds(exc: urllib.error.HTTPError, attempt: int) -> float:
    """Compute backoff delay for a 429/5xx response, honouring Retry-After."""
    # Prefer the server-supplied Retry-After (seconds or HTTP-date) if present.
    try:
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if retry_after is not None:
            try:
                return min(60.0, max(1.0, float(retry_after)))
            except ValueError:
                pass
        # OpenRouter sometimes returns a JSON body with reset hints.
        body = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        if body:
            data = json.loads(body) if body.strip().startswith("{") else {}
            for key in ("retry_after", "wait_seconds"):
                value = data.get(key) if isinstance(data, dict) else None
                if value is not None:
                    try:
                        return min(60.0, max(1.0, float(value)))
                    except (TypeError, ValueError):
                        pass
    except Exception:
        pass
    # Default: exponential backoff capped at 60s.
    return min(60.0, 4.0 * (2 ** attempt))


def _parse_json_with_repair(text: str) -> dict:
    """Robustly extract a JSON object from an LLM response."""
    if not text:
        return {"raw": ""}

    # 1. Strip markdown fences if any.
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # 2. Direct parse.
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            return {"items": result}
    except json.JSONDecodeError:
        pass

    # 3. Greedy match: from first { to last } (handles trailing commentary).
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # 4. Try stripping trailing commas before } / ].
            repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    return {"raw": text}
