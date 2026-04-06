# ARIS Project - Comprehensive Analysis and Documentation

**Last Updated:** March 29, 2026  
**Project Version:** 0.1.0  
**Python Requirement:** 3.11+

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Phase Structure](#phase-structure)
3. [Architecture & Directory Structure](#architecture--directory-structure)
4. [Core Modules (Phase 1 - Sealed)](#core-modules-phase-1---sealed)
5. [Graph Modules](#graph-modules)
6. [ML Modules (Phase 2 - Complete)](#ml-modules-phase-2---complete)
7. [Experimental Framework](#experimental-framework)
8. [Key Design Principles](#key-design-principles)
9. [Dependencies and Setup](#dependencies-and-setup)
10. [Usage Patterns](#usage-patterns)
11. [Testing Structure](#testing-structure)
12. [Data and Benchmarking](#data-and-benchmarking)

---

## Project Overview

**ARIS** (A Research Intelligence System) is a production-ready, deterministic Python system for research intelligence generation and analysis. It combines deterministic reasoning substrates (Phase 1) with optional ML-augmented extensions (Phase 2) for scientific document processing, entity/relation extraction, knowledge graph construction, and hypothesis generation.

### Key Characteristics

- **Deterministic by Design**: All Phase 1 operations are reproducible and falsifiable
- **Type-Safe**: Strict type hints throughout (mypy strict mode)
- **Modular**: Clear separation of concerns across 13+ modules
- **Lazy-Loading**: ML dependencies are optional and imported only when needed
- **Immutable Data**: No mutations of inputs; operations return new objects
- **Production-Ready**: Comprehensive error handling, logging, testing

### Project Goals

1. **Phase 1**: Deterministic research substrate with zero ML dependencies
2. **Phase 2**: ML-augmented extensions for entity/relation extraction and clustering
3. **Reproducibility**: Publication-quality benchmarking on peer-reviewed datasets (SciERC, DocRED)
4. **Falsifiability**: All claims traceable to explicit logic with no hidden heuristics

---

## Phase Structure

### Phase 1 (SEALED)
**Status**: Sealed December 16, 2025  
**Modules**: 0-9

Pure deterministic substrate with NO ML dependencies:
- Input validation and normalization
- Reasoning engine (template-based)
- Evaluation heuristics
- Memory persistence
- Tool abstraction
- Trace replay
- Document ingestion (text + PDF via optional pypdf)
- Knowledge graph construction
- Research planning
- Evidence-backed linking

**Constraints**:
- No randomness, no floating-point instability
- No autonomy loops or self-modification
- No learning or online parameter optimization
- All inputs remain unchanged (immutability)
- No non-deterministic libraries

### Phase 2 (COMPLETE)
**Status**: Complete and integrated  
**Modules**: 10-13 + 10B/11/12 (Novel Components)

ML-augmented extensions with SciBERT, HDBSCAN, spaCy:
- **Module 10**: SciBERT role induction (sentence-level role classification)
- **Module 10B**: Cross-document entity extraction (novel, token-level with disambiguation)
- **Module 11**: HDBSCAN concept clustering (ontology induction)
- **Module 12**: spaCy SRL semantic relations (context-aware, discourse patterns)
- **Module 13**: Hypothesis induction (graph motif detection + hypothesis impact scoring)

All Phase 2 tools:
- Return structured JSON error payloads if dependencies missing
- Execute deterministically with fixed random seeds
- Generate hypotheses only (no graph mutation)
- Include full provenance and metadata

---

## Architecture & Directory Structure

```
aris/
├── core/              # Phase 1 modules (0-9)
│   ├── input_interface.py        # Module 0: Input validation
│   ├── reasoning_engine.py       # Module 2: Template-based reasoning
│   ├── evaluation.py             # Module 3: Heuristic scoring
│   ├── tool.py                   # Module 4: Tool abstraction
│   ├── memory_store.py           # Core: Memory persistence
│   ├── run_loop.py               # Core: Orchestration
│   ├── trace_replay.py           # Module 5: Trace reconstruction
│   ├── comparative_runner.py     # Module 6: Multi-config comparison
│   └── logging_config.py         # Core: Deterministic logging
│
├── graph/             # Knowledge graph construction
│   ├── document_ingestion.py     # Module 7: Doc loading + packetization
│   ├── knowledge_graph.py        # Module 8: Graph construction + linking
│   ├── hypothesis_induction.py   # Module 13: Graph motif detection
│   ├── hypothesis_impact.py      # Module 13: Hypothesis impact scoring
│   └── research_planner.py       # Module 9: Research action planning
│
└── ml/                # Phase 2 ML extensions (opti onal)
    ├── entity_extraction.py      # Module 10B: Cross-doc entity extraction
    ├── role_induction.py         # Module 10: SciBERT role classification
    ├── ontology_induction.py     # Module 11: HDBSCAN clustering
    └── relation_induction.py     # Module 12: spaCy SRL relations

tests/
├── core/              # Unit tests for Phase 1
├── graph/             # Unit tests for graph modules
└── ml/                # Unit tests for ML modules

experiments/
├── evaluation_engine.py        # Benchmarking harness
├── system_wrappers.py          # Wrapper interfaces for SciERC/DocRED
└── utils/
    ├── datasets.py             # Dataset loading and validation
    └── plotting.py             # Evaluation visualization

data/
├── scierc/                      # SciERC dataset (multi-task: entities, relations, coreference)
│   ├── train.json
│   ├── dev.json
│   └── test.json
└── docred/                      # DocRED dataset (document-level relation extraction)
    ├── train_annotated.json
    ├── dev.json
    └── test.json

Scripts:
├── download_datasets.py         # Automated dataset acquisition
├── run_benchmarks.py            # Full evaluation pipeline
├── test_pipeline.py             # Quick benchmarking tests
├── test_data_loading.py         # Dataset integrity checks
├── check_dataset.py             # Dataset existence validation
├── demo_integrated_pipeline.py  # Phase 2 component integration demo
├── debug_*.py                   # Debugging utilities
```

---

## Core Modules (Phase 1 - Sealed)

### Module 0 & 1: Input Interface (`input_interface.py`)

**Purpose**: Strict input validation, normalization, and packet creation

**Key Classes**:
- `InputError`: Base exception for validation failures
- `InputValidationError`: Validation constraint violations
- `EmptyInputError`: Rejects empty/whitespace-only input
- `MaxLengthExceededError`: Enforces 10K character limit (configurable)
- `MissingSourceError`: Source identifier required

**InputPacket** (frozen dataclass):
```python
@dataclass(frozen=True)
class InputPacket:
    text: str                              # Normalized input (trimmed, \n canonical)
    source: str                            # Traceability identifier
    request_id: uuid.UUID                  # UUIDv4, deterministic
    timestamp: datetime                    # UTC timestamp
```

**InputInterface** (deterministic validator):
- `accept(raw_text, source)`: Validates, normalizes newlines, generates packet
- Constraints: non-empty, max 10K chars, valid UTF-8
- Newline canonicalization: ensures cross-platform determinism

**Design Notes**:
- No I/O, no randomness, fully deterministic
- Immutable packet prevents downstream mutation
- Source required for safety-critical traceability

---

### Module 2: Reasoning Engine (`reasoning_engine.py`)

**Purpose**: Generate explicit reasoning steps from input (deterministic or LLM-ready)

**Key Classes**:
- `ReasoningResult` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class ReasoningResult:
      reasoning_steps: list[str]       # 3-5 explicit reasoning steps
      confidence_score: float          # [0.0, 1.0]
      input_packet: InputPacket        # Original input reference
  ```

**ReasoningEngine**:
- `reason(input_packet) -> ReasoningResult`
- **Current Implementation**: Template-based (placeholder)
  - Step 1: Request metadata
  - Step 2-3: Input characteristics (char count, word count)
  - Step 4+: Conditional steps based on input length
  - Confidence: Based on input length (0.2-0.9 heuristic)
- **Future-Ready**: API designed for LLM-based replacement without breaking changes

**Design Notes**:
- Deterministic output for deterministic reasoning
- Confidence scoring based on input characteristics only
- Future-proof design for LLM integration

---

### Module 3: Evaluation System (`evaluation.py`)

**Purpose**: Score reasoning quality across 4 dimensions

**Key Classes**:
- `EvaluationResult` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class EvaluationResult:
      step_count_score: float          # Optimal 3-5 steps
      coherence_score: float           # Token overlap between steps
      confidence_alignment_score: float # Confidence ≈ expected from input length
      input_coverage_score: float      # Input terms present in steps
      overall_score: float             # Weighted average of above
  ```

**Evaluator**:
- `evaluate(reasoning_result) -> EvaluationResult`
- **Metrics**:
  - Step Count Score: Penalizes |count - 4|, favors 3-5 steps
  - Coherence Score: Non-empty step ratio
  - Confidence Alignment: Delta between confidence and expected from length
  - Input Coverage: % of input tokens present in reasoning steps
  - Overall Score: Unweighted average of 4 scores

**Design Notes**:
- Purely heuristic, no learned models
- Deterministic scoring
- No mutation of input reasoning result

---

### Module 4: Tool Abstraction (`tool.py`)

**Purpose**: Define stateless, deterministic string-to-string tools

**Key Classes**:
- `Tool` (ABC):
  ```python
  class Tool(ABC):
      @property
      @abstractmethod
      def name(self) -> str: ...
      
      @abstractmethod
      def execute(self, input_text: str) -> str: ...
  ```

**Built-in Tools**:
- `EchoTool`: Identity function
- `WordCountTool`: Count words in input

**ToolRegistry**:
- `register(tool: Tool)`: Register by name
- `get(name: str) -> Tool | None`: Retrieve by name
- `list_tools() -> list[str]`: List all registered names

**Design Notes**:
- No inference, no automatic tool selection
- Explicit register/get/list pattern
- Tools execute only when explicitly requested in `run_loop()`

---

### Memory Persistence (`memory_store.py`)

**Purpose**: Store and retrieve reasoning traces persistently

**Key Classes**:
- `MemoryTrace` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class MemoryTrace:
      request_id: uuid.UUID
      input_data: dict[str, str]           # InputPacket as dict
      reasoning_data: dict[str, object]    # reasoning_steps, confidence_score
      created_at: datetime
      evaluation_data: dict[str, float] | None = None
      tool_calls: list[dict[str, str]] | None = None
  ```

**MemoryStore** (ABC):
- `store(trace: MemoryTrace)`: Persist trace
- `retrieve(request_id: uuid.UUID) -> MemoryTrace | None`
- `list_all() -> list[MemoryTrace]`
- `clear()`: Clear all traces

**Implementations**:
- `InMemoryStore`: Thread-safe dict-based storage
- `FileBackedStore`: JSON file persistence with folder structure

**Design Notes**:
- Thread-safe implementations
- Immutable traces prevent mutation
- Append-only, no modification of stored traces

---

### Orchestration (`run_loop.py`)

**Purpose**: Process validated input packets deterministically

**Main Function**:
```python
def run_loop(
    input_packets: Sequence[InputPacket],
    reasoning_engine: ReasoningEngine,
    memory_store: MemoryStore,
    tool_registry: ToolRegistry | None = None,
    tool_calls: dict[int, list[tuple[str, str]]] | None = None,
) -> list[ReasoningResult]:
```

**Execution Flow**:
1. For each input packet:
   - Run reasoning engine
   - Create MemoryTrace
   - Optionally execute tool calls
   - Store trace in memory
   - Print conclusion (last reasoning step)
   - Return result

**Design Notes**:
- Deterministic control flow
- All dependencies injected (no global state)
- Tool execution optional and explicit

---

### Module 5: Trace Replay (`trace_replay.py`)

**Purpose**: Reconstruct execution flow from stored traces without recomputation

**Key Classes**:
- `ReplayFrame` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class ReplayFrame:
      frame_index: int
      frame_type: Literal["input", "reasoning_step", "evaluation", "tool_call"]
      content: dict[str, object]  # Event-specific data
  ```

- `TraceReplay` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class TraceReplay:
      request_id: uuid.UUID
      frames: list[ReplayFrame]  # Ordered execution flow
      created_at: datetime
  ```

**ReplayEngine**:
- `replay(trace: MemoryTrace) -> TraceReplay`
- **Frame Order**: input → reasoning_steps → evaluation → tool_calls
- **Features**:
  - No recomputation, uses stored data only
  - Backward compatible (skips missing sections gracefully)
  - Immutable frame sequence

**Design Notes**:
- Useful for post-hoc analysis without re-execution
- No mutation of source trace
- All frames immutable

---

### Module 6: Comparative Runner (`comparative_runner.py`)

**Purpose**: Execute same input across multiple system configurations

**Key Classes**:
- `SystemConfig` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class SystemConfig:
      name: str
      reasoning_engine: ReasoningEngine
      evaluator: Evaluator | None = None
      tool_registry: ToolRegistry | None = None
      tool_calls: dict[int, list[tuple[str, str]]] | None = None
  ```

- `ComparativeResult` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class ComparativeResult:
      input_packet: InputPacket
      traces_by_config: dict[str, MemoryTrace | Exception]  # Failure isolation
      created_at: datetime
  ```

**ComparativeRunner**:
- `run(input_packet, configs) -> ComparativeResult`
- **Features**:
  - Independent MemoryTrace per configuration
  - Exception isolation (failure in one config doesn't affect others)
  - No aggregation or merging of results

**Design Notes**:
- Useful for A/B testing variant strategies
- Each configuration tested independently
- Exceptions captured per config, not propagated

---

## Graph Modules

### Module 7: Document Ingestion (`document_ingestion.py`)

**Purpose**: Load and canonicalize documents from various formats

**Key Classes**:
- `Document` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class Document:
      content: str                       # Normalized text
      source: str                        # File path or identifier
      format: str                        # "text", "pdf", etc.
      metadata: dict[str, str]          # File size, page count, etc.
      document_id: uuid.UUID            # Unique identifier
      created_at: datetime              # Ingestion timestamp
  ```

- `DocumentCorpus` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class DocumentCorpus:
      documents: list[Document]
      corpus_id: uuid.UUID
      created_at: datetime
  ```

**Loaders**:
- `PlainTextLoader`:
  - Loads UTF-8 text files
  - Rejects empty documents
  - Canonicalizes line endings (\r\n → \n)
  
- `PDFLoader`:
  - Requires optional pypdf dependency
  - Extracts text from all pages
  - Rejects empty PDFs
  - Graceful error if pypdf unavailable

**DocumentIngestor**:
- `ingest(file_path, format) -> Document`
- Format-specific loading logic

**CorpusPacketizer**:
- Corpus → validated InputPacket objects
- No automatic chunking
- No auto-detection, explicit format required

**Design Notes**:
- No chunking, no inference, no auto-detection
- Immutable Document objects
- Comprehensive metadata tracking
- Graceful handling of optional dependencies

---

### Module 8: Knowledge Graph (`knowledge_graph.py`)

**Purpose**: Build cross-document knowledge graphs with deterministic linking

**Key Classes**:
- `Node` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class Node:
      node_id: uuid.UUID
      document_id: uuid.UUID
      label: str                         # Document title, entity name, etc.
      node_type: str                     # "document", "entity", "concept", etc.
      metadata: dict[str, str]
  ```

- `Edge` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class Edge:
      edge_id: uuid.UUID
      source_id: uuid.UUID               # Node ID
      target_id: uuid.UUID               # Node ID
      edge_type: str                     # "cites", "references", "supports", etc.
      evidence: str                      # Text justification
      reasoning_trace_id: uuid.UUID      # Linked to reasoning trace
      confidence: float                  # [0.0, 1.0] from evaluation
      metadata: dict[str, str]
      created_at: datetime
  ```

- `KnowledgeGraph` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class KnowledgeGraph:
      graph_id: uuid.UUID
      nodes: list[Node]
      edges: list[Edge]
      created_at: datetime
  ```

- `LinkCandidate` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class LinkCandidate:
      source_document_id: uuid.UUID
      target_document_id: uuid.UUID
      link_type: str
      hint: str                          # Optional hint for reasoning
      metadata: dict[str, str]
  ```

**Linker**:
- `generate_candidates(corpus, strategy, ...)` → LinkCandidate list
- **Strategies**:
  - `keyword_overlap`: Find docs sharing specified keyword
  - `metadata_match`: Find docs with matching metadata field
  - `sequential`: Link consecutive documents in corpus

**LinkMaterializer**:
- `materialize(candidates, graph)`: Create Edge objects with evidence and confidence
- Uses ReasoningEngine for evidence justification
- Uses Evaluator for confidence scoring

**Design Notes**:
- Only deterministic heuristics, no embeddings
- Cross-document linking without semantic similarity
- Evidence-backed links tied to reasoning traces
- Immutable graph and nodes/edges

---

### Module 9: Research Planner (`research_planner.py`)

**Purpose**: Propose next research actions without executing them

**Key Classes**:
- `ResearchAction` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class ResearchAction:
      action_id: uuid.UUID               # Deterministic UUID
      action_type: str                   # "investigate_gap", "resolve_contradiction",
                                         # "strengthen_evidence"
      description: str                   # Human-readable action
      evidence: str                      # Concrete evidence motivating action
      rationale: str                     # Why action is useful
      priority: float                    # [0.0, 1.0]
      metadata: dict[str, str]           # No IDs required
  ```

- `PlannerContext` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class PlannerContext:
      graph: KnowledgeGraph              # Not mutated
      strategy: str                      # "gap-driven" | "contradiction-driven" |
                                         # "weak-evidence"
      max_actions: int = 10
  ```

**ResearchPlanner**:
- `plan(context) -> list[ResearchAction]`
- **Strategies**:
  - `gap-driven`: Identify low-connectivity nodes/documents
  - `contradiction-driven`: Find conflicting edges
  - `weak-evidence`: Identify edges with low confidence
- **Features**:
  - Deterministic ranking (priority desc, then action_type, description, action_id)
  - No execution, only proposals
  - Full evidence and rationale attached

**Design Notes**:
- Policy-bounded (no autonomy loops)
- All actions have explicit evidence
- No graph mutation
- Deterministic sorting ensures reproducibility

---

### Module 13: Hypothesis Induction (`hypothesis_induction.py`)

**Purpose**: Detect graph motifs and generate testable research hypotheses

**Key Classes**:
- `HypothesisCandidate` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class HypothesisCandidate:
      hypothesis_id: uuid.UUID
      hypothesis_type: str               # "gap", "cluster", "chain", "hub", "contradiction"
      description: str                   # Observable pattern statement
      supporting_nodes: tuple[uuid.UUID, ...]
      supporting_edges: tuple[uuid.UUID, ...]
      confidence: float                  # [0.0, 1.0] pattern strength
      metadata: dict[str, Any]           # Pattern-specific data
      created_at: datetime
  ```

**HypothesisInductionEngine**:
- `induce_hypotheses(graph) -> list[HypothesisCandidate]`
- **Hypothesis Types**:
  - `gap`: Nodes with shared neighbors but no direct connection
  - `cluster`: Highly interconnected node groups
  - `chain`: Sequential chains of connections
  - `hub`: Nodes with many connections (high degree)
  - `contradiction`: Conflicting edge types between same node pair
- **Features**:
  - No outcome prediction
  - No graph modification
  - Deterministic behavior
  - Pattern-specific metadata
  - Configurable min_confidence threshold

**Design Notes**:
- Purely descriptive (no predictions)
- Graph immutability verified via assertions
- Closed set of hypothesis types
- Each type has explicit detection logic

---

### Module 13 Extended: Hypothesis Impact Scoring (`hypothesis_impact.py`)

**Purpose**: Rank hypotheses by research potential using entity and relation signals

**Key Classes**:
- `HypothesisImpactScore` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class HypothesisImpactScore:
      hypothesis_id: str
      hypothesis_type: str
      impact_score: float                # Overall research value [0.0, 1.0]
      novelty_signal: float              # New connections [0.0, 1.0]
      citation_signal: float             # Entity mention frequency [0.0, 1.0]
      relation_signal: float             # Research relation involvement [0.0, 1.0]
      base_confidence: float             # Original hypothesis confidence
      total_signal: float                # Sum of normalized signals
      provenance: dict[str, Any]         # Weights, factors, etc.
  ```

**HypothesisImpactScoringTool**:
- Extends `Tool` interface
- Input: Hypotheses + entity mentions + research relations (JSON)
- Output: Ranked HypothesisImpactScore list (JSON)
- **Signals**:
  - Novelty: Based on hypothesis type weights (gap=0.9, contradiction=1.0, hub=0.6)
  - Citation: Entity mention frequency across documents
  - Relation: Strength of relations connecting hypothesis nodes (OUTPERFORMS=1.0, SUPPORTS=0.75)
- **Algorithm**:
  - Compute 3 signals per hypothesis
  - Scale by base confidence
  - Rank by impact_score descending

**Design Notes**:
- Synthesizes entity + relation extraction for impact assessment
- Immutability verified via snapshot comparison
- Full metadata provenance included

---

## ML Modules (Phase 2 - Complete)

### Module 10: SciBERT Role Induction (`role_induction.py`)

**Purpose**: Classify research text spans into semantic roles using SciBERT

**Key Classes**:
- `RoleCandidate` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class RoleCandidate:
      span_text: str                     # Text span
      start: int                         # Character offset
      end: int                           # Character offset
      role: str                          # "method", "result", "finding", etc.
      confidence: float                  # [0.0, 1.0]
      provenance: dict[str, Any]         # Model, tokenizer, strategy
  ```

**SciBertRoleTool** (extends `Tool`):
- Model: `allenai/scibert_scivocab_uncased`
- **Default Role Vocabulary**: method, result, finding, limitation, background, objective, conclusion
- **Input**: Raw text
- **Output**: JSON list of RoleCandidate dicts
- **Algorithm**:
  - Load SciBERT tokenizer + model (lazy-loaded)
  - Split text into sentences
  - For each span: Create template "[MASK] role: {span}"
  - Compute logits over role vocabulary
  - Select highest-probability role
  - Confidence from softmax probability
- **Error Handling**:
  - Missing dependencies → structured JSON error payload
  - Includes install hint: `pip install aris[ml]`
  - Exception details in output

**Design Notes**:
- Hypothesis generation only (no graph mutation)
- Fixed random seed (7) for reproducibility
- Full provenance includes model, tokenizer, strategy
- Device-agnostic (CPU/GPU support via parameter)

---

### Module 10B: Cross-Document Entity Extraction (`entity_extraction.py`)

**Purpose**: Extract fine-grained scientific entities with cross-document context

**Novel Contributions**:
- Token-level extraction (vs. Module 10 sentence-level)
- Cross-document context awareness for disambiguation
- Entity mention frequency tracking across collection
- Confidence propagation based on contextual similarity

**Key Classes**:
- `EntityCandidate` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class EntityCandidate:
      span: str
      start: int                         # Character offset
      end: int                           # Character offset
      entity_type: str                   # "METHOD", "DATASET", "METRIC", "TASK", ...
      confidence: float                  # [0.0, 1.0]
      context_score: float               # Cross-doc similarity [0.0, 1.0]
      mention_count: int                 # Frequency across corpus
      provenance: dict[str, Any]         # Extraction metadata
  ```

**ScientificEntityTool** (extends `Tool`):
- Model: `allenai/scibert_scivocab_uncased`
- **Entity Types**: METHOD, DATASET, METRIC, TASK, MATERIAL, RESULT
- **Input** (JSON):
  ```json
  {
    "text": "document text",
    "document_id": "optional_id",
    "context_documents": [
      {"text": "...", "id": "..."},
      ...
    ]
  }
  ```
- **Output**: JSON list of EntityCandidate dicts
- **Algorithm**:
  - Pattern-based baseline extraction (regex per entity type)
  - SciBERT embeddings for context vectors
  - Cross-document context matching for disambiguation
  - Confidence from pattern + context similarity
  - Mention frequency tracking across documents

**Pattern Examples**:
- **METHOD**: BERT, GPT, ResNet, Transformer, CNN, RNN, Attention, etc.
- **DATASET**: ImageNet, CIFAR, MNIST, SQuAD, GLUE, etc.
- **METRIC**: BLEU, ROUGE, F1, accuracy, precision, recall, etc.
- **TASK**: NER, sentiment analysis, question answering, etc.

**Design Notes**:
- Novel approach: token-level with cross-doc context
- Cross-document entity memory tracks seen contexts
- Immutability enforced via snapshot comparison
- Full entity mention statistics included

---

### Module 11: HDBSCAN Ontology Clustering (`ontology_induction.py`)

**Purpose**: Cluster text items using embeddings and HDBSCAN for ontology induction

**Key Classes**:
- `OntologyClusterCandidate` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class OntologyClusterCandidate:
      cluster_id: int
      member_ids: list[str]              # IDs of items in cluster
      representative_id: str             # Central item ID
      representative_text: str           # Central item text
      size: int
      stability: float                   # HDBSCAN stability metric [0.0, 1.0]
      provenance: dict[str, Any]         # Clustering metadata
  ```

**OntologyClusteringTool** (extends `Tool`):
- Model: `allenai/scibert_scivocab_uncased` for embeddings
- **Input** (JSON): `[{"id": "str", "text": "str"}, ...]`
- **Output**: JSON list of OntologyClusterCandidate dicts
- **HDBSCAN Params**:
  - min_cluster_size: 2
  - min_samples: 1
  - metric: euclidean
  - cluster_selection_epsilon: 0.0
  - cluster_selection_method: "eom" (excess of mass)
- **Algorithm**:
  1. Embed all texts with SciBERT
  2. Compute distance matrix
  3. Run HDBSCAN clustering
  4. Extract clusters with stability scores
  5. Find representative item (closest to centroid) per cluster

**Design Notes**:
- Hypothesis generation only (no ontology materialization)
- Deterministic with fixed seed (11)
- Full metadata includes model, params, stability
- Immutability enforced via JSON snapshot

---

### Module 12: spaCy Semantic Relations (`relation_induction.py`)

**Purpose**: Extract semantic relations with discourse patterns using spaCy SRL

**Key Classes**:
- `RelationCandidate` (frozen dataclass):
  ```python
  @dataclass(frozen=True)
  class RelationCandidate:
      relation_type: str
      subject_span: str
      subject_start: int                 # Character offset
      subject_end: int                   # Character offset
      object_span: str
      object_start: int                  # Character offset
      object_end: int                    # Character offset
      predicate: str                     # Intermediate predicate
      confidence: float                  # [0.0, 1.0]
      provenance: dict[str, Any]         # Extraction metadata
  ```

**SemanticRelationTool** (extends `Tool`):
- Model: `en_core_web_sm` (spaCy)
- **Relation Ontology** (closed set):
  - CAUSES, PART_OF, RELATED_TO, USES, PRODUCES
  - DEPENDS_ON, CONTRADICTS, ACHIEVES, OUTPERFORMS, IMPROVES, SUPPORTS
- **Input** (JSON):
  ```json
  {
    "text": "document text",
    "entities": [
      {"span": "text", "start": int, "end": int, "type": "entity_type"},
      ...
    ]
  }
  ```
- **Output**: JSON list of RelationCandidate dicts
- **Algorithm**:
  - Parse text with spaCy NLP pipeline
  - Extract SRL frames (subject/predicate/object)
  - Map predicates to closed relation ontology
  - Compute confidence from semantic similarity
  - Include discourse patterns (e.g., context around predicates)

**Design Notes**:
- Context-aware using discourse patterns
- Closed ontology prevents unbounded output
- Deterministic with fixed seed (12)
- Full provenance includes strategy and model
- Immutability enforced via JSON snapshot

---

## Experimental Framework

### Core Benchmarking Infrastructure (`experiments/`)

**Purpose**: Production-quality evaluation on peer-reviewed datasets (SciERC, DocRED)

**Key Components**:

#### 1. **evaluation_engine.py**
- `EvaluationEngine` class
- Computes metrics per configuration:
  - Entity-level: Precision, Recall, F1
  - Relation-level: Precision, Recall, F1
  - Macro averages across datasets
- Generates comparison tables and visualizations

#### 2. **system_wrappers.py**
- Unified interface for different systems:
  - ARIS (native)
  - SciBERT (via transformers)
  - spaCy (via spacy pipelines)
  - OpenIE (via open-source toolkit)
- **EntityPrediction**:
  ```python
  @dataclass(frozen=True)
  class EntityPrediction:
      span: str
      start: int
      end: int
      entity_type: str
      confidence: float
  ```
- **RelationPrediction**:
  ```python
  @dataclass(frozen=True)
  class RelationPrediction:
      subject_span: str
      object_span: str
      relation_type: str
      confidence: float
  ```

#### 3. **utils/datasets.py**
- **DatasetNotFoundError**: Explicit failure if datasets missing
- `validate_datasets(data_path)`: Check presence of required datasets
- `load_scierc_entities(split, data_path)`:
  - Loads SciERC entity annotations
  - Returns list of documents with entities
- `load_scierc_relations(split, data_path)`:
  - Loads SciERC relation annotations
  - Includes coreference chains
- `load_docred_relations(split, data_path)`:
  - Loads DocRED relation annotations
  - Document-level relation extraction

**Design Philosophy**: NO FABRICATED DATA, NO SILENT FALLBACKS. Hard errors on missing datasets.

#### 4. **utils/plotting.py**
- Visualization functions:
  - `plot_entity_precision_recall_curves()`: PR curves per entity type
  - `plot_entity_best_f1_bars()`: F1 comparison bars
  - `plot_relation_pr_curves()`: PR curves for relations
  - `plot_relation_f1_breakdown()`: F1 breakdown by relation type
  - `plot_relation_prediction_coverage()`: Coverage analysis
  - `generate_summary_tables()`: Latex/CSV tables

---

### Benchmarking Pipeline (`run_benchmarks.py`)

**Entry Point**: `python run_benchmarks.py --dataset scierc --dataset-path <path> --output experiments/`

**Workflow**:
1. **Setup**: Create output directory structure (logs, figures, tables)
2. **Validation**: Check datasets exist and are properly formatted
3. **Loading**: Load SciERC/DocRED gold standards
4. **System Evaluation**:
   - For each system in ["ARIS", "SciBERT", "spaCy", "OpenIE"]:
     - Extract entities on all documents
     - Extract relations on all documents
     - Compute metrics
5. **Analysis**:
   - Generate comparison tables (Latex, CSV)
   - Create visualizations (PR curves, F1 bars)
   - Write logs with timing and error info
6. **Output**:
   - `experiments/entity_eval/`: Entity evaluation results
   - `experiments/relation_eval/`: Relation evaluation results
   - `experiments/figures/`: Visualization plots
   - `experiments/tables/`: Summary tables
   - `experiments/logs/`: Execution logs

**Configuration**:
- SUPPORTED_DATASETS: {"scierc", "docred"}
- SUPPORTED_SYSTEMS: {"ARIS", "SciBERT", "spaCy", "OpenIE"}
- SUPPORTED_TASKS: {"entity", "relation"}

---

## Key Design Principles

### 1. **Determinism**
- All Phase 1 operations produce identical outputs for identical inputs
- No randomness, no floating-point non-determinism
- Fixed random seeds in Phase 2 tools
- Deterministic UUID generation in Phase 1

### 2. **Falsifiability**
- Every claim (reasoning step, score, action) traceable to explicit logic
- No hidden heuristics, no magic constants
- All parameters documented and configurable
- Clear evidence backing all produced artifacts

### 3. **Immutability**
- All inputs remain unchanged after processing
- Operations return new objects, never mutate inputs
- Frozen dataclasses prevent accidental mutation
- No side effects beyond append-only traces

### 4. **Type Safety**
- Strict type hints throughout (mypy strict mode)
- No `Any` except for lazy-loaded dependencies
- All function signatures explicit
- Runtime validation confirms type expectations

### 5. **Separation of Concerns**
- Clear module boundaries (core, graph, ml)
- Each module does one thing well
- Minimal inter-module dependencies
- Tool abstraction for extensibility

### 6. **Lazy Loading**
- All ML dependencies optional
- Imported only when tools invoked
- Structured error payloads if missing
- Install hints provided automatically

### 7. **Reproducibility**
- All operations logged deterministically
- Full trace persistence for post-hoc analysis
- Trace replay without recomputation
- Publication-quality benchmarking infrastructure

---

## Dependencies and Setup

### Base Installation (Phase 1 only)
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Unix/macOS

pip install -e .
```

**Phase 1 Dependencies**: None beyond Python 3.11 stdlib

### With ML Extensions (Phase 2)
```bash
pip install -e ".[ml]"
```

**Phase 2 Dependencies**:
- torch>=2.1.0,<3.0.0
- transformers>=4.37.0,<5.0.0
- numpy>=1.24
- scikit-learn>=1.3
- hdbscan>=0.8.36
- spacy>=3.7.0

### With PDF Support
```bash
pip install -e ".[pdf]"
```

**PDF Dependencies**:
- pypdf>=4.0.0

### Development Setup
```bash
pip install -e ".[dev]"
```

**Development Dependencies**:
- mypy>=1.7.0
- ruff>=0.1.0
- pytest>=7.4.0

### All Extras
```bash
pip install -e ".[ml,pdf,dev]"
```

### Verification
```bash
# Check Phase 1 works
python -m aris

# Check dataset pipeline
python test_pipeline.py

# Run full benchmarks
python run_benchmarks.py --dataset scierc --dataset-path e:/aris/data --output experiments/
```

---

## Usage Patterns

### Pattern 1: Basic Input → Reasoning → Evaluation
```python
from aris.core.input_interface import InputInterface
from aris.core.reasoning_engine import ReasoningEngine
from aris.core.evaluation import Evaluator
from aris.core.memory_store import InMemoryStore
from aris.core.run_loop import run_loop

# Setup
input_interface = InputInterface()
reasoning_engine = ReasoningEngine()
evaluator = Evaluator()
memory_store = InMemoryStore()

# Process input
packet = input_interface.accept("Your text here", source="cli")
result = reasoning_engine.reason(packet)
evaluation = evaluator.evaluate(result)

# Store trace
trace = MemoryTrace.from_reasoning_result(result, evaluation)
memory_store.store(trace)

# Retrieve and inspect
stored = memory_store.retrieve(packet.request_id)
```

### Pattern 2: Tool Execution
```python
from aris.core.tool import ToolRegistry, EchoTool, WordCountTool
from aris.core.run_loop import run_loop

# Setup registry
registry = ToolRegistry()
registry.register(EchoTool())
registry.register(WordCountTool())

# Define tool calls (index 0 = first input packet)
tool_calls = {
    0: [
        ("echo", "test"),
        ("word_count", "hello world"),
    ]
}

# Execute with tools
results = run_loop(
    [packet],
    reasoning_engine,
    memory_store,
    tool_registry=registry,
    tool_calls=tool_calls,
)
```

### Pattern 3: Document Ingestion → Knowledge Graph
```python
from aris.graph.document_ingestion import PlainTextLoader, DocumentCorpus
from aris.graph.knowledge_graph import Linker, LinkMaterializer

# Load documents
loader = PlainTextLoader()
docs = [
    loader.load("doc1.txt"),
    loader.load("doc2.txt"),
    loader.load("doc3.txt"),
]
corpus = DocumentCorpus(documents=docs, corpus_id=uuid4(), created_at=datetime.now(UTC))

# Generate link candidates
linker = Linker()
candidates = linker.generate_candidates(corpus, strategy="keyword_overlap", keyword="research")

# Materialize links into graph (using reasoning + evaluation)
materializer = LinkMaterializer(reasoning_engine, evaluator)
graph = materializer.materialize(candidates, corpus)
```

### Pattern 4: Hypothesis Generation from Graph
```python
from aris.graph.hypothesis_induction import HypothesisInductionEngine
from aris.graph.hypothesis_impact import HypothesisImpactScoringTool

# Induce hypotheses from graph
engine = HypothesisInductionEngine(min_confidence=0.5)
hypotheses = engine.induce_hypotheses(graph)

# Score by impact
scorer = HypothesisImpactScoringTool()
impact_json = json.dumps({
    "hypotheses": [h.to_dict() for h in hypotheses],
    "entity_mentions": entity_frequency_counts,
    "relations": extracted_relations,
})
impact_scores = json.loads(scorer.execute(impact_json))
```

### Pattern 5: ML Tool Usage
```python
from aris.ml.entity_extraction import ScientificEntityTool
from aris.ml.relation_induction import SemanticRelationTool

# Entity extraction
entity_tool = ScientificEntityTool()
input_data = {"text": "...", "document_id": "doc1"}
entities_json = entity_tool.execute(json.dumps(input_data))
entities = json.loads(entities_json)

# Relation extraction
relation_tool = SemanticRelationTool()
input_data = {
    "text": "...",
    "entities": [{"span": "BERT", "start": 0, "end": 4, "type": "METHOD"}, ...],
}
relations_json = relation_tool.execute(json.dumps(input_data))
relations = json.loads(relations_json)
```

### Pattern 6: Comparative Evaluation
```python
from aris.core.comparative_runner import ComparativeRunner, SystemConfig

# Setup multiple configurations
configs = [
    SystemConfig(
        name="baseline",
        reasoning_engine=ReasoningEngine(),
        evaluator=Evaluator(),
    ),
    SystemConfig(
        name="with-tools",
        reasoning_engine=ReasoningEngine(),
        evaluator=Evaluator(),
        tool_registry=registry,
        tool_calls={0: [("echo", "test")]},
    ),
]

# Run comparison
runner = ComparativeRunner()
result = runner.run(packet, configs)

# Inspect results
for config_name, trace_or_exception in result.traces_by_config.items():
    if isinstance(trace_or_exception, Exception):
        print(f"{config_name} failed: {trace_or_exception}")
    else:
        print(f"{config_name} trace: {trace_or_exception.request_id}")
```

---

## Testing Structure

### Test Organization
```
tests/
├── core/
│   ├── test_input_interface.py        # InputInterface, InputPacket
│   ├── test_reasoning_engine.py       # ReasoningEngine, ReasoningResult
│   ├── test_evaluation.py             # Evaluator, EvaluationResult
│   ├── test_tool.py                   # Tool, ToolRegistry
│   ├── test_memory_store.py           # InMemoryStore, MemoryTrace
│   ├── test_run_loop.py               # run_loop orchestration
│   ├── test_trace_replay.py           # ReplayEngine, TraceReplay
│   └── test_comparative_runner.py     # ComparativeRunner
│
├── graph/
│   ├── test_document_ingestion.py     # Loaders, Document, Corpus
│   ├── test_knowledge_graph_simple.py # Node, Edge, Graph basics
│   ├── test_knowledge_graph.py        # Linker, linking strategies
│   ├── test_hypothesis_induction.py   # HypothesisInductionEngine
│   ├── test_hypothesis_impact.py      # HypothesisImpactScoringTool (if exists)
│   └── test_research_planner.py       # ResearchPlanner, strategies
│
├── ml/
│   ├── test_entity_extraction.py      # ScientificEntityTool
│   ├── test_role_induction.py         # SciBertRoleTool
│   ├── test_ontology_induction.py     # OntologyClusteringTool
│   ├── test_relation_induction.py     # SemanticRelationTool
│   └── test_ml_optional_governance.py # Optional dependency handling
│
└── test_cli.py                        # CLI entrypoint testing
```

### Running Tests
```bash
# All tests
pytest

# Specific module
pytest tests/core/

# With coverage
pytest --cov=aris

# Specific test
pytest tests/core/test_input_interface.py::test_input_validation
```

### Test Patterns
- **Immutability**: Verify inputs unchanged after processing
- **Determinism**: Same input → same output
- **Error Handling**: Validate exception messages and types
- **Type Safety**: Check return types match signatures
- **Edge Cases**: Empty inputs, boundary values, malformed data
- **Optional Deps**: Verify graceful errors when ML libs missing

---

## Data and Benchmarking

### Dataset Acquisition

#### SciERC (Multi-Task Extraction)
- **Source**: https://github.com/allenai/scierc
- **Citation**: Luan et al., EMNLP 2018
- **Structure**:
  - ~350 training documents
  - ~50 dev documents
  - ~100 test documents
- **Tasks**: Entity extraction, relation extraction, coreference
- **Expected Location**: `data/scierc/{train,dev,test}.json`

#### DocRED (Document-Level Relation Extraction)
- **Source**: https://github.com/thunlp/DocRED
- **Citation**: Yao et al., ACL 2019
- **Structure**:
  - ~3,000 training documents
  - ~1,000 dev documents
  - ~1,000 test documents
- **Task**: Document-level relation extraction
- **Expected Location**: `data/docred/{train_annotated,dev,test}.json`

### Automated Download
```bash
python download_datasets.py --output e:/aris/data
# or individually
python download_datasets.py --output e:/aris/data --dataset scierc
python download_datasets.py --output e:/aris/data --dataset docred
```

### Dataset Validation
```bash
python test_pipeline.py  # Quick validation
python check_dataset.py  # Full integrity check
```

### Benchmarking
```bash
# Full pipeline
python run_benchmarks.py \
    --dataset scierc \
    --dataset-path e:/aris/data \
    --output experiments/

# This produces:
# - experiments/entity_eval/: Entity evaluation results
# - experiments/relation_eval/: Relation evaluation results
# - experiments/figures/: Visualizations
# - experiments/tables/: Summary tables (Latex, CSV)
# - experiments/logs/: Execution logs
```

### Evaluation Metrics
- **Entity Level**: Precision, Recall, F1 (per type and macro)
- **Relation Level**: Precision, Recall, F1 (per type and macro)
- **System Comparison**: Side-by-side tables, PR curves, F1 bars
- **Coverage**: What % of gold annotations each system extracts

---

## Future Enhancement Points

### Phase 1 Extensions
- [ ] Additional document formats (DOCX, HTML, Markdown)
- [ ] Configurable reasoning templates
- [ ] Pluggable evaluation metrics
- [ ] Database-backed MemoryStore (PostgreSQL, MongoDB)
- [ ] Distributed run_loop (multiple workers)

### Phase 2 Extensions
- [ ] Fine-tuned SciBERT on custom entity types
- [ ] Domain-specific relation ontologies
- [ ] Active learning for ambiguous hypotheses
- [ ] Reinforcement learning for planning

### Benchmarking
- [ ] More datasets (ACE, NYT, TACL)
- [ ] Confidence calibration analysis
- [ ] Cross-domain generalization
- [ ] Efficiency profiling (inference time, memory)

### Integration
- [ ] REST API for remote execution
- [ ] Graph database backend (Neo4j)
- [ ] Interactive UI for hypothesis exploration
- [ ] Paper pipeline (ingestion → extraction → analysis)

---

## Module Dependency Graph

```
Module 0 (Input Interface)
    ↓
Module 2 (Reasoning Engine)
    ↓
Module 3 (Evaluation System)
    ↓
Module 5 (Trace Replay) ← Memory Store
                         ← Tool Abstraction (Module 4)
Module 6 (Comparative Runner)

Module 7 (Document Ingestion)
    ↓
Module 8 (Knowledge Graph)
    ├→ Module 9 (Research Planner)
    ├→ Module 13 (Hypothesis Induction)
    └→ Module 13 Extended (Hypothesis Impact)

Module 10, 10B, 11, 12 (ML Tools - optional)
    ↓
Demo Integration (demo_integrated_pipeline.py)
    ↓
Benchmarking (run_benchmarks.py)
```

---

## Quick Reference: File Locations

| Purpose | File |
|---------|------|
| Entry point | `aris/__main__.py` |
| Input validation | `aris/core/input_interface.py` |
| Deterministic reasoning | `aris/core/reasoning_engine.py` |
| Quality scoring | `aris/core/evaluation.py` |
| Tool execution | `aris/core/tool.py` |
| Trace persistence | `aris/core/memory_store.py` |
| Orchestration | `aris/core/run_loop.py` |
| Trace analysis | `aris/core/trace_replay.py` |
| Multi-config comparison | `aris/core/comparative_runner.py` |
| Document loading | `aris/graph/document_ingestion.py` |
| Graph construction | `aris/graph/knowledge_graph.py` |
| Hypothesis generation | `aris/graph/hypothesis_induction.py` |
| Hypothesis ranking | `aris/graph/hypothesis_impact.py` |
| Action proposals | `aris/graph/research_planner.py` |
| Role classification | `aris/ml/role_induction.py` |
| Entity extraction | `aris/ml/entity_extraction.py` |
| Relation extraction | `aris/ml/relation_induction.py` |
| Clustering | `aris/ml/ontology_induction.py` |
| Benchmarking | `run_benchmarks.py` |
| System wrappers | `experiments/system_wrappers.py` |
| Evaluation engine | `experiments/evaluation_engine.py` |
| Dataset utilities | `experiments/utils/datasets.py` |
| Visualization | `experiments/utils/plotting.py` |

---

## Maintenance Notes

### Configuration & Constants

Key configurable values:
- **Input max length**: 10,000 characters (InputInterface)
- **Optimal step count**: 3-5 steps (Evaluator)
- **Min hypothesis confidence**: 0.5 (HypothesisInductionEngine, configurable)
- **Planning max actions**: 10 (ResearchPlanner, configurable)
- **Entity types**: METHOD, DATASET, METRIC, TASK, MATERIAL, RESULT (closed)
- **Relation ontology**: 11 types (CAUSES, PART_OF, RELATED_TO, ... - closed)
- **Hypothesis types**: 5 types (gap, cluster, chain, hub, contradiction - closed)

### Random Seeds
- Module 10 (SciBertRoleTool): seed=7
- Module 10B (ScientificEntityTool): seed=42
- Module 11 (OntologyClusteringTool): seed=11
- Module 12 (SemanticRelationTool): seed=12

### Common Debugging Steps
1. Check Phase 1 works: `python -m aris --file test.txt`
2. Check dataset validity: `python check_dataset.py`
3. Check ML dependencies: `python test_pipeline.py`
4. Check traces: Use `TraceReplay` to inspect stored MemoryTrace objects
5. Check type hints: `mypy aris/`
6. Check linting: `ruff check aris/`

### Adding New Features
1. Define immutable dataclass for results
2. Implement deterministic logic (no randomness)
3. Add comprehensive type hints
4. Include full provenance/metadata
5. Write unit tests (immutability, determinism)
6. Update this documentation
7. Run full test suite: `pytest`
8. Run linting: `ruff check . && mypy aris/`

---

**This document serves as the single source of truth for ARIS project architecture, functionality, and usage. Update this document for all future changes to maintain a complete reference.**

