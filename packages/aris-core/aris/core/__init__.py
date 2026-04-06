"""Phase 1 Core: Deterministic research intelligence substrate (Modules 1-6, 0).

This package provides the foundational deterministic infrastructure for ARIS:
- Input validation and request lifecycle
- Template-based reasoning and heuristic evaluation
- Tool abstraction and memory trace persistence
- Deterministic comparative experimentation

No external ML dependencies. Reproducible. Falsifiable.
"""

from aris.core.comparative_runner import ComparativeResult, ComparativeRunner, SystemConfig
from aris.core.evaluation import EvaluationResult, Evaluator
from aris.core.input_interface import InputError, InputInterface, InputPacket
from aris.core.logging_config import setup_logging
from aris.core.memory_store import FileBackedMemoryStore, InMemoryStore, MemoryStore, MemoryTrace
from aris.core.reasoning_engine import ReasoningEngine, ReasoningResult
from aris.core.tool import EchoTool, Tool, ToolRegistry, WordCountTool
from aris.core.trace_replay import ReplayEngine, ReplayFrame, TraceReplay
from aris.core.run_loop import run_loop

__all__ = [
    "InputInterface",
    "InputPacket",
    "InputError",
    "ReasoningEngine",
    "ReasoningResult",
    "Evaluator",
    "EvaluationResult",
    "Tool",
    "EchoTool",
    "WordCountTool",
    "ToolRegistry",
    "MemoryTrace",
    "MemoryStore",
    "InMemoryStore",
    "FileBackedMemoryStore",
    "ReplayFrame",
    "TraceReplay",
    "ReplayEngine",
    "SystemConfig",
    "ComparativeResult",
    "ComparativeRunner",
    "run_loop",
    "setup_logging",
]
