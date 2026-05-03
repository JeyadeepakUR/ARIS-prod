"""Groq LLM provider — free tier, fastest inference available.

Free tier: 14,400 requests/day on Llama 3.3 70B.
Sign up at https://console.groq.com to get an API key.
"""
from __future__ import annotations

import json
import re

from groq import Groq


class GroqProvider:
    def __init__(self, *, model: str = "llama-3.3-70b-versatile", api_key: str) -> None:
        self._client = Groq(api_key=api_key)
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
        text = self.complete(
            prompt + "\n\nRespond with valid JSON only. No markdown, no explanation.",
            max_tokens=max_tokens,
            temperature=0.0,
        )
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"raw": text}
