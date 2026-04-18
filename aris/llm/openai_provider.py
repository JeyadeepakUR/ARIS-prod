"""OpenAI LLM provider for ARIS."""

from __future__ import annotations

import json
import re


class OpenAIProvider:
    """Calls the OpenAI chat completions API."""

    def __init__(self, *, model: str, api_key: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required: pip install openai"
            ) from exc
        self._client = OpenAI(api_key=api_key or None)
        self._model = model

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

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
