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

## Module 0: Completed Components

- **InputInterface**: Accepts raw text, trims whitespace, attaches `UUIDv4` request_id and UTC `timestamp`, returns `InputPacket`.
- **ReasoningEngine**: Deterministic, template-based reasoning producing 3–5 explicit steps and a heuristic confidence (0–1). API is future-proof for LLM internals.
- **MemoryStore**: Thread-safe abstraction with two implementations:
	- In-memory store for tests
	- File-backed JSON store persisting full traces by `request_id`
	Schema:
	```json
	{
		"request_id": "...",
		"input": { ... },
		"reasoning": { ... },
		"created_at": "..."
	}
	```
- **Run Loop**: Deterministic `run_loop()` that wires InputInterface → ReasoningEngine → MemoryStore and prints a final conclusion per input.

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
│   ├── __init__.py         # Package initialization
│   ├── __main__.py         # Entry point for `python -m aris`
│   └── logging_config.py   # Deterministic logging configuration
```

## Design Principles

- **Zero unnecessary dependencies**: Only stdlib for core functionality
- **Strict type hints**: Full mypy strict mode compliance
- **Deterministic logging**: Consistent, reproducible log output
- **Clear module boundaries**: Well-defined separation of concerns
- **Production-ready**: Clean architecture from day one
