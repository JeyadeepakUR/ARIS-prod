"""
Tests for the ARIS CLI entrypoint.

Simulates stdin and file inputs without touching real files.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

import aris.__main__ as cli


def test_cli_processes_stdin(monkeypatch: pytest.MonkeyPatch, capsys: object) -> None:
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("hello"))

    cli.main([])

    output = capsys.readouterr().out.strip().splitlines()  # type: ignore[attr-defined]
    assert len(output) == 1
    assert "Single-word input detected" in output[0]


def test_cli_processes_multiple_files(monkeypatch: pytest.MonkeyPatch, capsys: object) -> None:
    contents = {
        "file1.txt": "hello",
        "file2.txt": "another test",
    }

    def fake_read_text(self: Path, encoding: str = "utf-8") -> str:  # noqa: ANN001
        if str(self) in contents:
            return contents[str(self)]
        raise FileNotFoundError(str(self))

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    cli.main(["--file", "file1.txt", "file2.txt"])

    output = capsys.readouterr().out.strip().splitlines()  # type: ignore[attr-defined]
    assert len(output) == 2
    assert all(line for line in output)


def test_cli_maps_input_errors(monkeypatch: pytest.MonkeyPatch, capsys: object) -> None:
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("   \n   "))

    with pytest.raises(SystemExit) as exc:
        cli.main([])

    assert exc.value.code == 2
    err = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "empty" in err.lower()


def test_cli_maps_file_errors(monkeypatch: pytest.MonkeyPatch, capsys: object) -> None:
    def fake_read_text(self: Path, encoding: str = "utf-8") -> str:  # noqa: ANN001
        raise FileNotFoundError("missing")

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    with pytest.raises(SystemExit) as exc:
        cli.main(["--file", "missing.txt"])

    assert exc.value.code == 3
    err = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "missing" in err.lower()
