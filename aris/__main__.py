"""Command-line entrypoint for deterministic ARIS execution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aris.input_interface import InputError, InputInterface
from aris.memory_store import InMemoryStore
from aris.reasoning_engine import ReasoningEngine
from aris.run_loop import run_loop

__all__ = [
    "main",
    "_main",
    "_read_inputs",
    "_parse_args",
    "sys",
]


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process text input through ARIS.")
    parser.add_argument(
        "--file",
        dest="files",
        nargs="+",
        type=Path,
        help="One or more UTF-8 text files to process",
    )
    return parser.parse_args(argv)


def _read_inputs(files: list[Path] | None) -> list[tuple[str, str]]:
    if files:
        inputs: list[tuple[str, str]] = []
        for path in files:
            text = path.read_text(encoding="utf-8")
            inputs.append((text, str(path)))
        return inputs

    return [(sys.stdin.read(), "stdin")]


def _main(argv: list[str]) -> None:
    args = _parse_args(argv)

    input_interface = InputInterface()
    reasoning_engine = ReasoningEngine()
    memory_store = InMemoryStore()

    raw_inputs = _read_inputs(args.files)
    packets = [input_interface.accept(text, source=source) for text, source in raw_inputs]

    run_loop(packets, reasoning_engine, memory_store)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments, route inputs, and manage exit codes."""

    argv = sys.argv[1:] if argv is None else argv

    try:
        _main(argv)
    except InputError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(2) from exc
    except (OSError, UnicodeDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(3) from exc
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(5) from exc


if __name__ == "__main__":
    main()
