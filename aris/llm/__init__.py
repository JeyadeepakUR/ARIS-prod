"""ARIS LLM provider abstraction layer.

Use get_provider() to obtain the configured backend.
"""

from aris.llm.provider import LLMProvider, get_provider

__all__ = ["LLMProvider", "get_provider"]
