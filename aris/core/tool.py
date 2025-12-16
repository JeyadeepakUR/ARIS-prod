"""Stateless tool abstraction for deterministic ARIS operations."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Tool(ABC):
    """Abstract base class for deterministic string-to-string tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""

    @abstractmethod
    def execute(self, input_text: str) -> str:
        """Execute the tool on input, returning output string.

        Args:
            input_text: Input string (tool receives as-is, no preprocessing)

        Returns:
            Output string (no length guarantees or transformations)
        """


class EchoTool(Tool):
    """Echo tool that returns input unchanged."""

    @property
    def name(self) -> str:
        return "echo"

    def execute(self, input_text: str) -> str:
        return input_text


class WordCountTool(Tool):
    """Tool that counts words in input."""

    @property
    def name(self) -> str:
        return "word_count"

    def execute(self, input_text: str) -> str:
        count = len(input_text.split())
        return str(count)


class ToolRegistry:
    """Explicit tool registration and lookup."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by name.

        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Retrieve a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not registered
        """
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return sorted(self._tools.keys())
