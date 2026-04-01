"""SciBERT-based role induction tool (Module 10).

This module provides hypothesis-generation only. It does not assert correctness or
materialize graph nodes. All outputs are deterministic and reproducible.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from aris.core.tool import Tool

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer


@dataclass(frozen=True)
class RoleCandidate:
    """Span-level role hypothesis produced by the SciBERT role tool."""

    span_text: str
    start: int
    end: int
    role: str
    confidence: float
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping with clamped confidence for safety."""

        safe_confidence = min(max(self.confidence, 0.0), 1.0)
        return {
            "span_text": self.span_text,
            "start": self.start,
            "end": self.end,
            "role": self.role,
            "confidence": safe_confidence,
            "provenance": self.provenance,
        }


class SciBertRoleTool(Tool):
    """Role induction using a fixed SciBERT masked language model."""

    MODEL_NAME = "allenai/scibert_scivocab_uncased"
    DEFAULT_ROLE_VOCAB: tuple[str, ...] = (
        "method",
        "result",
        "finding",
        "limitation",
        "background",
        "objective",
        "conclusion",
    )

    def __init__(self, role_vocab: Iterable[str] | None = None, device: str | None = None) -> None:
        self._role_vocab: tuple[str, ...] = tuple(role_vocab) if role_vocab is not None else self.DEFAULT_ROLE_VOCAB
        self._device = device or "cpu"
        # Stored as Any to avoid hard dependency on HF type stubs
        self._tokenizer: Any = None
        self._model: Any = None
        self._seed = 7

    @property
    def name(self) -> str:
        return "scibert_role_induction"

    def execute(self, input_text: str) -> str:
        text = input_text or ""
        if not text.strip():
            return "[]"

        try:
            candidates = self._infer_roles(text)
            payload = [candidate.to_dict() for candidate in candidates]
            return json.dumps(payload, ensure_ascii=True, sort_keys=True)
        except Exception as exc:  # noqa: BLE001
            error_payload = {
                "error": str(exc),
                "error_type": exc.__class__.__name__,
                "model": self.MODEL_NAME,
            }
            return json.dumps(error_payload, ensure_ascii=True, sort_keys=True)

    def _infer_roles(self, text: str) -> list[RoleCandidate]:
        self._ensure_model()
        sentences = self._split_sentences(text)
        cursor = 0
        candidates: list[RoleCandidate] = []

        for sentence in sentences:
            start = text.find(sentence, cursor)
            if start == -1:
                start = text.find(sentence)
                if start == -1:
                    continue
            end = start + len(sentence)
            span_text = text[start:end]
            role, confidence = self._score_span(span_text)
            provenance = {
                "model": self.MODEL_NAME,
                "tokenizer": self.MODEL_NAME,
                "mask_token": self._tokenizer.mask_token if self._tokenizer else "",
                "role_vocab": list(self._role_vocab),
                "strategy": "mlm_mask_softmax",
            }
            candidates.append(
                RoleCandidate(
                    span_text=span_text,
                    start=start,
                    end=end,
                    role=role,
                    confidence=confidence,
                    provenance=provenance,
                )
            )
            cursor = end

        return candidates

    def _score_span(self, span_text: str) -> tuple[str, float]:
        assert self._tokenizer is not None
        assert self._model is not None

        # Deteministic template with a single mask slot
        template = f"This span describes the {self._tokenizer.mask_token} role: {span_text}"
        encoding = self._tokenizer(template, return_tensors="pt", truncation=True)
        mask_positions = (encoding.input_ids == self._tokenizer.mask_token_id).nonzero(as_tuple=True)
        if len(mask_positions[0]) == 0:
            return self._role_vocab[0], 0.0

        mask_index = mask_positions[1][0].item()

        # Lazy import to avoid hard dependency errors during module import
        import torch

        with torch.no_grad():
            outputs = self._model(**encoding.to(self._device))
            logits = outputs.logits[0, mask_index]
            role_token_ids: list[int] = []
            for role in self._role_vocab:
                tokens = self._tokenizer.tokenize(role)
                if not tokens:
                    continue
                role_token_ids.append(self._tokenizer.convert_tokens_to_ids(tokens[0]))

            if not role_token_ids:
                return self._role_vocab[0], 0.0

            role_logits = torch.tensor([logits[role_id].item() for role_id in role_token_ids])
            probs = torch.softmax(role_logits, dim=0)
            best_idx = int(torch.argmax(probs).item())
            return self._role_vocab[best_idx], float(probs[best_idx].item())

    def _ensure_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        # Lazy imports keep module importable even when dependencies are missing.
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("SciBERT dependencies (torch, transformers) are required for inference") from exc

        torch.manual_seed(self._seed)
        random.seed(self._seed)
        try:  # pragma: no cover - backend availability may vary
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModelForMaskedLM.from_pretrained(self.MODEL_NAME)
        self._model.to(self._device)
        self._model.eval()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]
