"""OpenRouter LLM provider for ARIS.

OpenRouter exposes an OpenAI-compatible API at https://openrouter.ai/api/v1.
Free-tier models (suffixed ':free') require no payment and count against the
free daily quota instead.

Default model: meta-llama/llama-4-scout:free  (free, no credit card required)
"""

from __future__ import annotations

import json
import re
import urllib.request


_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


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

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        payload = json.dumps({
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode()

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
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())

        return body["choices"][0]["message"]["content"] or ""

    def complete_json(self, prompt: str, *, max_tokens: int = 512) -> dict:
        full_prompt = prompt + "\n\nRespond with valid JSON only. Do not include markdown code fences."
        text = self.complete(full_prompt, max_tokens=max_tokens, temperature=0.1)
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"raw": text}
