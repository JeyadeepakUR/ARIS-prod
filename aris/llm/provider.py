"""LLM provider abstraction for ARIS intelligence layer.

Providers implement the LLMProvider protocol.
Use get_provider() to obtain the configured instance.

Supported backends: openai | anthropic | ollama | mock
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface every LLM backend must implement."""

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        """Return a completion string for the given prompt."""
        ...

    def complete_json(self, prompt: str, *, max_tokens: int = 512) -> dict:
        """Return a parsed JSON object from the LLM response."""
        ...


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response string."""
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw": text}


def get_provider(
    provider_name: str,
    *,
    model: str = "",
    api_key: str = "",
    base_url: str = "",
) -> LLMProvider:
    """Return an LLMProvider instance for the given backend name."""

    name = provider_name.strip().lower()

    if name == "openai":
        from aris.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(model=model or "gpt-4o-mini", api_key=api_key)

    if name == "anthropic":
        from aris.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model or "claude-3-5-haiku-20241022", api_key=api_key)

    if name == "ollama":
        from aris.llm.ollama_provider import OllamaProvider
        return OllamaProvider(model=model or "llama3.2", base_url=base_url or "http://localhost:11434")

    from aris.llm.mock_provider import MockProvider
    return MockProvider()
