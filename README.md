# ARIS

A clean, production-ready Python system with deterministic logging, strict type hints, and clear module boundaries.

## Phase Status

**Phase 1 (SEALED)**: Deterministic research intelligence substrate (Modules 0–9). No ML dependencies. Reproducible. Falsifiable. See [PHASE_1_CONTRACT.md](PHASE_1_CONTRACT.md).

**Phase 2 (COMPLETE)**: ML-augmented extensions (Modules 10–13). SciBERT role induction, HDBSCAN concept clustering, spaCy relation extraction, and deterministic hypothesis induction. Fully integrated. Isolated in `aris.ml`.

## Requirements

- Python 3.11 or higher

## Setup

Create a virtual environment and install the project:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Unix/macOS

# Base install (Phase 1 only, minimal dependencies):
pip install -e .

# With Phase 2 ML modules (SciBERT, HDBSCAN, spaCy):
pip install -e ".[ml]"

# With PDF support (Module 7):
pip install -e ".[pdf]"

# Development (type checking, linting, testing):
pip install -e ".[dev]"

# All extras:
pip install -e ".[ml,pdf,dev]"
```

All ML dependencies are **optional** and imported lazily. If invoked without required packages, Phase 2 tools return a structured error payload indicating missing dependencies and how to enable them.

## Usage

Run the system:

```bash
python -m aris
```

### Phase 2 Demo

Extract and analyze research papers with ML modules:

```bash
python demo_phase2.py paper.pdf --format pdf
```

This demo:
1. Ingests a PDF document (Module 7)
2. Extracts researcher roles via SciBERT (Module 10)
3. Clusters concepts using HDBSCAN (Module 11)
4. Extracts relations using spaCy SRL (Module 12)
5. Induces hypotheses from graph patterns (Module 13)

Example output:
```
MODULE 10: 293 conceptual role candidates extracted
MODULE 11: 40 clusters detected (21 cluster, 10 chain patterns)
MODULE 13: 31 hypotheses induced (21 cluster-type, 10 chain-type)
MODULE 12: 4 relations extracted (subject-predicate-object triples)
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

### Module 9: Research Planner
- **ResearchAction**: Frozen action proposal with deterministic UUID, type, evidence, rationale, priority [0.0, 1.0]
- **PlannerContext**: Frozen context holding graph, strategy, max_actions limit
- **ResearchPlanner**: Stateless planner with three explicit strategies:
  - Gap-driven: identifies isolated/weakly-connected nodes
  - Contradiction-driven: detects conflicting edge types between node pairs
  - Weak-evidence refinement: finds low-confidence or sparse-evidence edges
- Deterministic ranking by priority desc → action_type → description → action_id
- Guarantees: no execution, no mutation, no autonomy loops, no learning/optimization

### Module 10: Researcher Role Induction (SciBERT)
- **RoleCandidate**: Immutable candidate holding extracted span, role label (author/researcher/institution/etc.), position, confidence
- **RoleInductionEngine**: Uses pretrained SciBERT model to classify researcher roles in text
- Lazily imports torch/transformers; returns structured errors if ML extras not installed
- Conceptual filtering: excludes generic/background roles, retains domain-specific roles
- Deterministic with frozen random seeds

### Module 11: Concept Clustering (HDBSCAN)
- **ConceptCandidate**: Immutable cluster member with text, embedding, cluster_id, score
- **OntologyClusteringEngine**: Clusters concepts via HDBSCAN on semantic embeddings
- Generates deterministic UUIDs for clusters (uuid5 from member set)
- Extracts key terms from role spans (regex + spaCy fallback) to focus on concepts, not names
- Returns cluster hierarchy with silhouette scores
- Lazily imports scikit-learn/hdbscan; returns structured errors if ML extras not installed

### Module 12: Relation Extraction (spaCy SRL)
- **RelationCandidate**: Immutable relation holding subject, predicate, object, span positions, confidence
- **RelationInductionEngine**: Uses spaCy semantic role labeling to extract subject-predicate-object triples
- Overlap-based entity matching: robust to position variations in extracted spans
- Produces deterministic output with reproducible confidence scoring
- Lazily imports spacy; returns structured errors if ML extras not installed

### Module 13: Hypothesis Induction
- **HypothesisCandidate**: Immutable hypothesis with motif_type, concept_ids, confidence, rationale, evidence
- **HypothesisInductionEngine**: Induces **descriptive** hypotheses from graph patterns (not predictive)
- Five closed-set motif types:
  - **cluster**: Triangle (3+ concepts strongly connected)
  - **chain**: Linear path (A→B→C without C→A)
  - **gap**: Missing edge (A↔B, B↔C, but no A↔C)
  - **hub**: Hub-and-spoke (central node with radial edges)
  - **contradiction**: Conflicting edge types (A-supports-B and A-opposes-B)
- Minimum confidence threshold to filter low-signal hypotheses
- Deterministic UUIDs (uuid5) for reproducible output
- No ML, no mutation, pure graph pattern detection
- Guarantees: deterministic behavior, reproducible results, clear evidence for each hypothesis

### Module 0: Core Infrastructure
- **MemoryStore**: Thread-safe abstraction with two implementations:
	- In-memory store for tests
	- File-backed JSON store persisting full traces by `request_id`
- **run_loop**: Deterministic orchestrator wiring InputInterface → ReasoningEngine → MemoryStore
- CLI entry point with comprehensive error handling

## Try It

- **Phase 1 examples**:
	- Memory store demo: `python example_memory_store.py`
	- Thread-safety demo: `python example_thread_safety.py`
	- Reasoning engine demo: `python example_reasoning_engine.py`

- **Phase 2 end-to-end demo**:
	```bash
	python demo_phase2.py paper.pdf --format pdf
	```
	Requires `[ml]` and `[pdf]` extras installed.

- **Run all tests**:
	```bash
	python -m pytest tests/ -q
	```
	All 270+ tests passing (Phase 1 + Phase 2).

- **Type checking (strict mode)**:
	```bash
	python -m mypy aris --strict
	```

- **Lint (ruff)**:
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
├── pyproject.toml
├── aris/
│   ├── __init__.py, __main__.py, logging_config.py
│   ├── core/                      # Phase 1: Modules 0, 1-6
│   │   ├── __init__.py
│   │   ├── input_interface.py     # Module 1: Input validation
│   │   ├── reasoning_engine.py    # Module 2: Deterministic reasoning
│   │   ├── evaluation.py          # Module 3: Evaluation heuristics
│   │   ├── tool.py                # Module 4: Tool abstraction
│   │   ├── trace_replay.py        # Module 5: Trace replay
│   │   ├── comparative_runner.py  # Module 6: Comparative runner
│   │   ├── memory_store.py        # Module 0: Trace persistence
│   │   └── run_loop.py            # Module 0: Orchestration
│   ├── graph/                     # Phase 1: Modules 7-9 + Phase 2: Module 13
│   │   ├── __init__.py
│   │   ├── document_ingestion.py  # Module 7: Document ingestion
│   │   ├── knowledge_graph.py     # Module 8: Knowledge graph & linking
│   │   ├── research_planner.py    # Module 9: Research planning
│   │   └── hypothesis_induction.py # Module 13: Hypothesis induction
│   └── ml/                        # Phase 2: Modules 10-12 (optional)
│       ├── __init__.py
│       ├── role_induction.py      # Module 10: SciBERT role extraction
│       ├── ontology_clustering.py # Module 11: HDBSCAN concept clustering
│       └── relation_induction.py  # Module 12: spaCy SRL relation extraction
├── demo_phase2.py                 # End-to-end demo: PDF → roles → clusters → hypotheses
└── tests/
    ├── core/                      # Phase 1: Modules 0, 1-6 tests
    │   ├── test_input_interface.py
    │   ├── test_reasoning_engine.py
    │   ├── test_evaluation.py
    │   ├── test_tool.py
    │   ├── test_trace_replay.py
    │   ├── test_comparative_runner.py
    │   ├── test_memory_store.py
    │   └── test_run_loop.py
    ├── graph/                     # Phase 1: Modules 7-9 + Phase 2: Module 13
    │   ├── test_document_ingestion.py
    │   ├── test_knowledge_graph.py
    │   ├── test_knowledge_graph_simple.py
    │   ├── test_research_planner.py
    │   └── test_hypothesis_induction.py (19 tests, all passing)
    └── ml/                        # Phase 2: Modules 10-12 tests
        ├── test_role_induction.py
        ├── test_ontology_clustering.py
        └── test_relation_induction.py
```

**Governance**: Phase 1 is sealed. Phase 2 (Modules 10-13) is complete and fully integrated. See [PHASE_1_CONTRACT.md](PHASE_1_CONTRACT.md).
- **Strict type hints**: Full mypy strict mode compliance
- **Deterministic logging**: Consistent, reproducible log output
- **Clear module boundaries**: Well-defined separation of concerns
- **Production-ready**: Clean architecture from day one
- **Reproducible outputs**: All ML components use frozen random seeds (uuid5 for determinism)
- **Optional ML dependencies**: Phase 2 extras ([ml], [pdf]) are lazy-imported; graceful errors if missing
