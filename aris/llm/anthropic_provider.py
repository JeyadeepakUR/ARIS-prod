"""Anthropic Claude LLM provider for ARIS."""

from __future__ import annotations

import json
import re


class AnthropicProvider:
    """Calls the Anthropic Messages API."""

    def __init__(self, *, model: str, api_key: str) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required: pip install anthropic"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key or None)
        self._model = model

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        block = response.content[0]
        return getattr(block, "text", "") if response.content else ""

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
