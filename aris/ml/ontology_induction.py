"""Dynamic Ontology Induction (Module 11).

Embedding-based clustering strictly as hypothesis generation.
- No mutation of graphs
- No ontology materialization
- Deterministic behavior
- Full metadata logging
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable, TYPE_CHECKING

from aris.core.tool import Tool


class _MissingOptionalDependency(Exception):
    def __init__(self, required: set[str]):
        super().__init__("Missing optional dependency")
        self.required = required

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    import numpy as np


@dataclass(frozen=True)
class OntologyClusterCandidate:
    """Hypothesized ontology cluster produced from embeddings + HDBSCAN.

    The candidate is not a truth assertion; it is a proposal with metadata.
    """

    cluster_id: int
    member_ids: list[str]
    representative_id: str
    representative_text: str
    size: int
    stability: float
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "member_ids": list(self.member_ids),
            "representative_id": self.representative_id,
            "representative_text": self.representative_text,
            "size": self.size,
            "stability": float(max(0.0, min(1.0, self.stability))),
            "provenance": self.provenance,
        }


class OntologyClusteringTool(Tool):
    """Embeds texts with a fixed HF model, clusters with HDBSCAN, outputs candidates.

    Input schema (JSON string):
        [
          {"id": "str", "text": "str"},
          ...
        ]

    Output: JSON list of OntologyClusterCandidate mappings.
    """

    MODEL_NAME = "allenai/scibert_scivocab_uncased"
    HDBSCAN_PARAMS = {
        "min_cluster_size": 2,
        "min_samples": 1,
        "metric": "euclidean",
        "cluster_selection_epsilon": 0.0,
        "cluster_selection_method": "eom",
    }

    def __init__(self, device: str | None = None) -> None:
        self._device = device or "cpu"
        # Lazy-loaded heavy deps; stored as Any to avoid stub requirements
        self._tokenizer: Any = None
        self._model: Any = None
        self._seed = 11

    @property
    def name(self) -> str:
        return "ontology_clustering"

    def execute(self, input_text: str) -> str:
        text = input_text or ""
        if not text.strip():
            return "[]"
        try:
            items = self._parse_input(text)
            # Keep a snapshot for immutability checks
            snapshot = json.loads(json.dumps(items))
            candidates = self._cluster_items(items)
            # Ensure we didn't mutate inputs
            assert items == snapshot
            return json.dumps([c.to_dict() for c in candidates], ensure_ascii=True, sort_keys=True)
        except _MissingOptionalDependency as exc:
            return json.dumps(
                {
                    "error": "Missing optional dependency",
                    "module": "ontology_induction",
                    "required": sorted(list(exc.required)),
                    "install_hint": "pip install aris[ml]",
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        except Exception as exc:  # noqa: BLE001
            # Generic structured error that still exposes metadata
            return json.dumps(
                {
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                    "module": "ontology_induction",
                    "model": self.MODEL_NAME,
                    "hdbscan_params": self.HDBSCAN_PARAMS,
                },
                ensure_ascii=True,
                sort_keys=True,
            )

    def _parse_input(self, text: str) -> list[dict[str, str]]:
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Input must be a JSON list of {id,text} objects")
        items: list[dict[str, str]] = []
        for obj in data:
            if not isinstance(obj, dict):
                continue
            id_ = str(obj.get("id", "")).strip()
            txt = str(obj.get("text", "")).strip()
            if id_ and txt:
                items.append({"id": id_, "text": txt})
        if not items:
            return []
        return items

    def _cluster_items(self, items: list[dict[str, str]]) -> list[OntologyClusterCandidate]:
        if not items:
            return []
        # Ordering: for single-item inputs, surface embedding deps first;
        # for multi-item inputs, surface clustering deps first.
        if len(items) <= 1:
            try:
                embeddings = self._embed([x["text"] for x in items])
            except _MissingOptionalDependency as exc:
                raise exc
        else:
            self._ensure_clustering_deps()
            embeddings = self._embed([x["text"] for x in items])
        labels, probabilities, persistence = self._hdbscan_cluster(embeddings)

        # Build clusters
        id_list = [x["id"] for x in items]
        text_list = [x["text"] for x in items]
        clusters: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            if lab < 0:
                continue  # noise ignored
            clusters.setdefault(int(lab), []).append(idx)

        candidates: list[OntologyClusterCandidate] = []
        for cid in sorted(clusters.keys()):
            indices = clusters[cid]
            member_ids = [id_list[i] for i in indices]
            member_vecs = [embeddings[i] for i in indices]
            centroid = self._mean_vec(member_vecs)
            rep_idx_local = self._closest_index(centroid, member_vecs)
            rep_idx = indices[rep_idx_local]
            stability = float(persistence.get(cid, 0.0))
            provenance = {
                "model": self.MODEL_NAME,
                "tokenizer": self.MODEL_NAME,
                "hdbscan_params": dict(self.HDBSCAN_PARAMS),
                "probability_mean": float(sum(probabilities[i] for i in indices) / max(1, len(indices))),
            }
            candidates.append(
                OntologyClusterCandidate(
                    cluster_id=cid,
                    member_ids=member_ids,
                    representative_id=id_list[rep_idx],
                    representative_text=text_list[rep_idx],
                    size=len(indices),
                    stability=stability,
                    provenance=provenance,
                )
            )
        return candidates
    def _ensure_clustering_deps(self) -> None:
        try:
            import importlib
            importlib.import_module("numpy")  # type: ignore[import-not-found]
            importlib.import_module("hdbscan")  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise _MissingOptionalDependency({"hdbscan", "numpy"}) from exc

    # --- Embedding & clustering helpers ---

    def _ensure_model(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        try:
            import importlib
            torch = importlib.import_module("torch")  # type: ignore[import-not-found]
            transformers = importlib.import_module("transformers")  # type: ignore[import-not-found]
            AutoModel = getattr(transformers, "AutoModel")
            AutoTokenizer = getattr(transformers, "AutoTokenizer")
        except Exception as exc:  # pragma: no cover
            raise _MissingOptionalDependency({"torch", "transformers"}) from exc

        torch.manual_seed(self._seed)
        random.seed(self._seed)
        try:  # pragma: no cover
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = AutoModel.from_pretrained(self.MODEL_NAME)
        self._model.to(self._device)
        self._model.eval()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        import torch  # lazy

        all_vecs: list[list[float]] = []
        for txt in texts:
            enc = self._tokenizer(txt, return_tensors="pt", truncation=True)
            with torch.no_grad():
                out = self._model(**enc.to(self._device))
                last_hidden = out.last_hidden_state  # [1, seq, dim]
                attn = enc.attention_mask.to(self._device).unsqueeze(-1)
                summed = (last_hidden * attn).sum(dim=1)
                counts = attn.sum(dim=1).clamp(min=1)
                mean = summed / counts
                vec = mean.squeeze(0).tolist()
                all_vecs.append([float(x) for x in vec])
        return all_vecs

    def _hdbscan_cluster(self, embeddings: list[list[float]]) -> tuple[list[int], list[float], dict[int, float]]:
        try:
            import importlib
            np = importlib.import_module("numpy")  # type: ignore[import-not-found]
            hdbscan = importlib.import_module("hdbscan")  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise _MissingOptionalDependency({"hdbscan", "numpy"}) from exc

        X = np.array(embeddings, dtype=np.float32)  # type: ignore[attr-defined]
        if X.shape[0] == 0:
            return [], [], {}
        clusterer = hdbscan.HDBSCAN(**self.HDBSCAN_PARAMS)  # type: ignore[attr-defined]
        labels = clusterer.fit_predict(X)
        probs = getattr(clusterer, "probabilities_", None)
        probabilities = probs.tolist() if probs is not None else [1.0] * len(labels)
        persistence = {}
        pers = getattr(clusterer, "cluster_persistence_", None)
        if pers is not None:
            for idx, val in enumerate(pers):
                persistence[int(idx)] = float(val)
        return labels.tolist(), probabilities, persistence

    @staticmethod
    def _mean_vec(vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        sums = [0.0] * dim
        for v in vectors:
            for i in range(dim):
                sums[i] += v[i]
        return [s / max(1, len(vectors)) for s in sums]

    @staticmethod
    def _closest_index(target: list[float], vectors: list[list[float]]) -> int:
        def dist2(a: list[float], b: list[float]) -> float:
            return sum((ai - bi) * (ai - bi) for ai, bi in zip(a, b))

        best_i, best_d = 0, math.inf
        for i, v in enumerate(vectors):
            d = dist2(target, v)
            if d < best_d:
                best_i, best_d = i, d
        return best_i
