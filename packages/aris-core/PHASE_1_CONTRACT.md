# ARIS Phase 1 Contract: Sealed

**Status**: SEALED December 16, 2025  
**Scope**: Modules 0–9 (Deterministic substrate)  
**Next Phase**: Phase 2 (ML extensions) — awaiting user signal

## Governance Principles

### Reproducibility
- All Phase 1 operations are deterministic and produce identical outputs for identical inputs
- No randomness, no floating-point instability, no non-deterministic libraries
- Reasoning, evaluation, and planning are fully reproducible

### Falsifiability
- Every claim (reasoning step, evaluation score, action priority) can be traced to explicit logic
- No hidden heuristics, no implicit thresholds, no magic constants
- All parameters are documented and deterministic

### No ML Dependencies
- Phase 1 code depends **only** on Python 3.11+ stdlib (+ optional pypdf)
- No torch, transformers, sklearn, hdbscan, or any ML framework
- ML capabilities are exclusively Phase 2

### No Autonomy Loops
- Phase 1 does not execute its own planning decisions
- Research actions are proposed, not executed
- No self-modifying behavior

### No Learning or Optimization
- Phase 1 does not adapt based on results
- No parameter tuning, no gradient updates, no online learning
- Policies are static and explicit

### Immutability and Non-Mutation
- All inputs (documents, graphs, traces) remain unchanged
- Operations return new immutable objects
- No side effects on corpus or memory store beyond append-only traces

---

## Sealed Modules

### Module 0: Core Infrastructure
- `InputInterface`: Input validation (non-empty, max 10k chars, UTF-8)
- `InputPacket`: Immutable request wrapper with deterministic UUIDs, UTC timestamps
- `MemoryStore` (2 implementations): In-memory and file-backed JSON persistence
- `run_loop`: Orchestration without tool execution
- `logging_config`: Deterministic log output

### Module 1: Input Interface
- `InputInterface.accept()`: Trimming, validation, deterministic request_id generation
- `InputPacket`: Frozen dataclass with request_id (UUIDv4), timestamp (UTC)

### Module 2: Reasoning Engine
- `ReasoningEngine.reason()`: Template-based deterministic reasoning steps (3–5 steps)
- `ReasoningResult`: Frozen with reasoning_steps, confidence_score, input_packet
- Confidence scoring based on input length only (no learned models)

### Module 3: Evaluation System
- `Evaluator.evaluate()`: 4-dimensional heuristic scoring:
  - Step count score (3–5 steps optimal)
  - Coherence score (token overlap)
  - Confidence alignment (confidence ≈ expected from input length)
  - Input coverage (input terms in reasoning steps)
- `EvaluationResult`: Frozen scores with overall weighted average

### Module 4: Tool Abstraction
- `Tool`: ABC for stateless string-to-string tools
- Built-in: `EchoTool`, `WordCountTool`
- `ToolRegistry`: Explicit register/get/list_tools (no inference)
- Tools execute only when explicitly requested in `run_loop(tool_calls)`

### Module 5: Trace Replay Engine
- `ReplayFrame`: Immutable record of event (input, reasoning_step, evaluation, tool_call)
- `TraceReplay`: Ordered sequence of frames from MemoryTrace
- `ReplayEngine`: Reconstruct execution flow without recomputation
- Backward compatible with old traces (gracefully skips absent sections)

### Module 6: Comparative Runner
- `SystemConfig`: Variant description (reasoning engine, evaluator, tools)
- `ComparativeResult`: Traces grouped by config name
- `ComparativeRunner`: Same InputPacket across multiple configs
- Independent MemoryTrace per config; failure isolation per config

### Module 7: Document Ingestion
- `Document`: Frozen dataclass with content, source, format, metadata, document_id
- `DocumentCorpus`: Collection with corpus_id
- `PlainTextLoader`: UTF-8 text + line-ending normalization
- `PDFLoader`: Optional (pypdf) PDF extraction
- `DocumentIngestor`: Format-specific ingestion
- `CorpusPacketizer`: Corpus → validated InputPacket objects
- No auto-detection, no inference, no chunking

### Module 8: Knowledge Graph
- `Node`: Frozen node with node_id, document_id, label, node_type, metadata
- `Edge`: Frozen edge with evidence, reasoning_trace_id, confidence, metadata
- `KnowledgeGraph`: Frozen graph of nodes and edges
- `LinkCandidate`: Proposed link (source, target, type, hint)
- `Linker`: Deterministic candidate generation:
  - keyword_overlap: case-insensitive keyword matching
  - metadata_match: field-value matching
  - sequential: consecutive ordering
- `LinkMaterializer`: Justify + evaluate candidates → Edge (confidence ≥ 0.5)
- Helper functions: `build_graph_from_corpus()`, `add_edges_to_graph()`

### Module 9: Research Planner
- `ResearchAction`: Frozen action proposal with evidence, rationale, priority [0.0, 1.0]
- `PlannerContext`: Frozen planning context with graph, strategy, max_actions
- `ResearchPlanner`: Three deterministic strategies:
  - Gap-driven: identify isolated/weakly-connected nodes
  - Contradiction-driven: detect conflicting edge types
  - Weak-evidence refinement: find low-confidence or sparse-evidence edges
- Deterministic ranking: priority desc → action_type → description → action_id
- No action execution; only proposal

---

## Constraints (Binding)

1. **Phase 1 code MUST NOT depend on ML libraries** (torch, sklearn, hdbscan, transformers, etc.)
2. **Phase 1 code MUST NOT import from `aris.ml`**
3. **Phase 1 behavior MUST remain unchanged** across experiments
4. **All Phase 1 operations must be deterministic and reproducible**
5. **No autonomy loops**: planning proposes, does not execute
6. **No learning**: no parameter adaptation, no optimization
7. **No inference**: no hidden heuristics, all logic is explicit
8. **Immutability**: inputs are never mutated; operations return new objects

---

## Future Changes to Phase 1

Only bug fixes and clarity improvements are permitted. Any change must justify:

- **Bug fix**: Data corruption, incorrect logic, security issue
- **Clarity**: Documentation, naming, refactoring for readability
- **NOT permitted**: Performance optimization, new strategies, new capabilities

---

## Phase 2 Preparation

Phase 2 will extend ARIS with ML capabilities exclusively through:

1. **Tool Interfaces**: ML models wrap as Phase 1 tools
2. **Hypothesis Proposal**: ML generates candidates; Phase 1 evaluates
3. **Reasoning + Evaluation Gates**: Phase 1 judges ML output before materialization
4. **Optional Activation**: Phase 2 is opt-in; Phase 1 remains unchanged if unloaded

### Scope (Phase 2, PENDING)

- **Module 10**: SciBERT role induction (task, method, dataset, metric, finding)
- **Module 11**: Domain ontology clustering (HDBSCAN)
- **Module 12**: Embedding-based hypothesis generation (semantic similarity)
- **Module 13**: Semantic role labeling (SRL relations)

### Guaranteed Isolation

- No Phase 1 module imports Phase 2
- Phase 2 activation is explicit (opt-in tool registration)
- Phase 1 tests remain unchanged and passing

---

## Directory Structure

```
aris/
├── core/              # Phase 1: Modules 0, 1-6
├── graph/             # Phase 1: Modules 7-9
├── ml/                # Phase 2: Modules 10-13 (empty, awaiting user signal)
└── __init__.py, __main__.py, logging_config.py, etc.

tests/
├── core/              # Phase 1: Modules 0, 1-6 tests
├── graph/             # Phase 1: Modules 7-9 tests
└── ml/                # Phase 2: Modules 10-13 tests (empty)
```

---

## Test Coverage

### Phase 1 (All Pass ✅)

**Core** (Modules 0, 1-6):
- test_input_interface.py: 11 tests
- test_reasoning_engine.py: 16 tests
- test_evaluation.py: 6 tests
- test_tool.py: 20 tests
- test_trace_replay.py: 30 tests
- test_comparative_runner.py: 21 tests
- test_memory_store.py: 22 tests
- test_run_loop.py: 3 tests

**Graph** (Modules 7-9):
- test_document_ingestion.py: 29 tests
- test_knowledge_graph.py: 40 tests
- test_knowledge_graph_simple.py: 14 tests
- test_research_planner.py: 5 tests

**Total Phase 1**: 233 tests, all passing

### Phase 2 (EMPTY, awaiting implementation)

---

## Signing Off

**Phase 1 is SEALED.**

This contract governs all Phase 1 code and commits. No modifications to Phase 1 without justification and explicit governance review.

Phase 2 awaits user signal to begin implementation.

---

*Sealed: December 16, 2025*
