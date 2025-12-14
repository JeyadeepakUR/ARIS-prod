# ARIS

A clean, production-ready Python system with deterministic logging, strict type hints, and clear module boundaries.

## Requirements

- Python 3.11 or higher

## Setup

Create a virtual environment and install the project:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Unix/macOS

pip install -e .

# For PDF support (Module 7):
pip install -e ".[pdf]"
```

## Usage

Run the system:

```bash
python -m aris
```

Expected output:
```
ARIS booted
```

## Modules

### Module 1: Input Interface
- **InputInterface**: Validates raw text input with strict requirements (non-empty, max 10k chars, valid UTF-8)
- **InputPacket**: Immutable dataclass with text, source, request_id (UUIDv4), and UTC timestamp
- **ValidationError**: Custom exception hierarchy for validation failures
- Deterministic trimming and validation with clear error messages

### Module 2: Reasoning Engine
- **ReasoningEngine**: Deterministic, template-based reasoning producing 3–5 explicit steps
- Heuristic confidence scoring (0–1 range) based on input complexity
- **ReasoningResult**: Immutable result containing reasoning steps, confidence, and input packet reference
- No randomness, no I/O, fully deterministic behavior

### Module 3: Evaluation System
- **EvaluationEngine**: Scores reasoning quality across 4 dimensions
- Metrics:
  - Step count score (3–5 steps optimal)
  - Coherence score (keyword/phrase overlap between steps)
  - Confidence alignment (confidence matches step count)
  - Input coverage (input terms present in reasoning)
- **EvaluationResult**: Immutable scores with overall weighted average

### Module 4: Tool Abstraction
- **Tool ABC**: Abstract base class for stateless string-to-string tools
- Built-in tools: EchoTool (identity), WordCountTool (word count)
- **ToolRegistry**: Explicit register/get/list_tools pattern (no inference)
- **run_loop** extended: Optional tool_registry and tool_calls parameters
- Tools execute only when explicitly requested; outputs recorded in memory traces

### Module 5: Trace Replay Engine
- **ReplayFrame**: Immutable record of a single event (input, reasoning_step, evaluation, tool_call)
- **TraceReplay**: Ordered sequence of frames reconstructed from MemoryTrace
- **ReplayEngine**: Reconstructs execution flow without recomputation
- Strict ordering: input → reasoning steps → evaluation → tool calls
- Gracefully skips absent sections (backward compatible with old traces)
- JSON-serializable output for visualization or export

### Module 6: Comparative Runner
- **SystemConfig**: Describes a system variant (reasoning engine, evaluator, tools)
- **ComparativeResult**: Groups traces by configuration name
- **ComparativeRunner**: Executes same InputPacket across multiple configurations
- Produces independent MemoryTrace objects per configuration
- Isolates failures per configuration (exceptions stored, not propagated)
- No automatic persistence, no trace merging, no aggregation
- Preserves deterministic ordering and input identity

### Module 7: Document Ingestion & Canonicalization
- **Document**: Immutable document with content and provenance metadata
- **DocumentCorpus**: Collection of documents with corpus ID
- **PlainTextLoader**: UTF-8 text file loader with line ending normalization
- **PDFLoader**: PDF text extraction (requires pypdf optional dependency)
- **DocumentIngestor**: Format-specific ingestion with explicit format parameter
- **CorpusPacketizer**: Converts corpus to validated InputPacket objects
- Batch ingestion with failure isolation
- Reuses InputInterface validation
- No auto-detection, no inference, no chunking

### Module 8: Knowledge Graph & Cross-Document Linking
- **Node / Edge / KnowledgeGraph**: Frozen dataclasses for immutable graph structures with explicit evidence on every edge
- **LinkCandidate**: Immutable proposed link holding source/target docs, type, hint, metadata
- **Linker**: Deterministic candidate generation (keyword overlap, metadata match, sequential); no embeddings or hidden heuristics
- **LinkMaterializer**: Runs ReasoningEngine + Evaluator to justify candidates, require evidence, score confidence, and reject low-score links without mutating the corpus
- **Helpers**: build_graph_from_corpus (nodes only) and add_edges_to_graph (immutable merge)
- Guarantees: deterministic behavior, evidence completeness, input immutability, graph integrity

### Module 0: Core Infrastructure
- **MemoryStore**: Thread-safe abstraction with two implementations:
	- In-memory store for tests
	- File-backed JSON store persisting full traces by `request_id`
- **run_loop**: Deterministic orchestrator wiring InputInterface → ReasoningEngine → MemoryStore
- CLI entry point with comprehensive error handling

## Try It

- Examples:
	- Memory store demo: `python example_memory_store.py`
	- Thread-safety demo: `python example_thread_safety.py`

- Run loop tests:
	```bash
	python -m pytest tests/test_run_loop.py -q
	```

- Type checking (strict):
	```bash
	python -m mypy aris --strict
	```

- Lint (ruff):
	```bash
	python -m ruff check .
	```

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Type checking:

```bash
mypy aris
```

Code linting:

```bash
ruff check aris
```

## Project Structure

```
aris/
├── pyproject.toml          # Project configuration and dependencies
├── aris/
│   ├── __init__.py                # Package initialization
│   ├── __main__.py                # Entry point for `python -m aris`
│   ├── input_interface.py         # Module 1: Input validation
│   ├── reasoning_engine.py        # Module 2: Deterministic reasoning
│   ├── evaluation.py              # Module 3: Reasoning quality scoring
│   ├── tool.py                    # Module 4: Tool abstraction layer
│   ├── trace_replay.py            # Module 5: Trace replay engine
│   ├── comparative_runner.py      # Module 6: Comparative runner
│   ├── document_ingestion.py      # Module 7: Document ingestion
│   ├── knowledge_graph.py         # Module 8: Knowledge graph & linking
│   ├── memory_store.py            # Trace persistence
│   ├── run_loop.py                # Orchestration loop
│   └── logging_config.py          # Deterministic logging configuration
└── tests/
    ├── test_input_interface.py
    ├── test_reasoning_engine.py
    ├── test_evaluation.py
    ├── test_tool.py
    ├── test_trace_replay.py
    ├── test_comparative_runner.py
    ├── test_document_ingestion.py
	├── test_knowledge_graph.py
	├── test_knowledge_graph_simple.py
    ├── test_memory_store.py
    └── test_run_loop.py
```

## Design Principles

- **Zero unnecessary dependencies**: Only stdlib for core functionality
- **Strict type hints**: Full mypy strict mode compliance
- **Deterministic logging**: Consistent, reproducible log output
- **Clear module boundaries**: Well-defined separation of concerns
- **Production-ready**: Clean architecture from day one
