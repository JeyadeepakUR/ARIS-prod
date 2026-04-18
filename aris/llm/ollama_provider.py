"""Ollama local LLM provider for ARIS (http://localhost:11434)."""

from __future__ import annotations

import json
import re


class OllamaProvider:
    """Calls a locally running Ollama instance via its REST API."""

    def __init__(self, *, model: str, base_url: str) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError("httpx is required for Ollama: pip install httpx") from exc

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        response = httpx.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        return str(response.json().get("response", ""))

    def complete_json(self, prompt: str, *, max_tokens: int = 512) -> dict:
        full_prompt = prompt + "\n\nRespond with valid JSON only. Do not include markdown fences."
        text = self.complete(full_prompt, max_tokens=max_tokens, temperature=0.1)
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"raw": text}
