"""SRL-Enhanced Relation Induction (Module 12).

Semantic relation extraction strictly as hypothesis generation.
- No graph edge creation
- Closed relation ontology
- Deterministic inference
- Full metadata logging
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from aris.core.tool import Tool


class _MissingOptionalDependency(Exception):
    def __init__(self, required: set[str]):
        super().__init__("Missing optional dependency")
        self.required = required


@dataclass(frozen=True)
class RelationCandidate:
    """Hypothesized semantic relation between entities.

    The candidate is a proposal with provenance, not a truth assertion.
    """

    relation_type: str
    subject_span: str
    subject_start: int
    subject_end: int
    object_span: str
    object_start: int
    object_end: int
    predicate: str
    confidence: float
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_type": self.relation_type,
            "subject_span": self.subject_span,
            "subject_start": self.subject_start,
            "subject_end": self.subject_end,
            "object_span": self.object_span,
            "object_start": self.object_start,
            "object_end": self.object_end,
            "predicate": self.predicate,
            "confidence": float(max(0.0, min(1.0, self.confidence))),
            "provenance": self.provenance,
        }


class SemanticRelationTool(Tool):
    """Extract semantic relations using spaCy SRL with a closed relation ontology.

    Input schema (JSON string):
        {
          "text": "document text",
          "entities": [
            {"span": "text", "start": int, "end": int, "type": "entity_type"},
            ...
          ]
        }

    Output: JSON list of RelationCandidate mappings.
    """

    MODEL_NAME = "en_core_web_sm"
    RELATION_ONTOLOGY: tuple[str, ...] = (
        "CAUSES",
        "PART_OF",
        "RELATED_TO",
        "USES",
        "PRODUCES",
        "DEPENDS_ON",
        "CONTRADICTS",
    )

    def __init__(self) -> None:
        self._nlp: Any = None
        self._seed = 12

    @property
    def name(self) -> str:
        return "semantic_relation_extraction"

    def execute(self, input_text: str) -> str:
        text = input_text or ""
        if not text.strip():
            return "[]"
        try:
            data = self._parse_input(text)
            snapshot = json.dumps(data, sort_keys=True)
            candidates = self._extract_relations(data)
            # Ensure immutability
            assert snapshot == json.dumps(data, sort_keys=True)
            return json.dumps([c.to_dict() for c in candidates], ensure_ascii=True, sort_keys=True)
        except _MissingOptionalDependency as exc:
            return json.dumps(
                {
                    "error": "Missing optional dependency",
                    "module": "relation_induction",
                    "required": sorted(list(exc.required)),
                    "install_hint": "pip install aris[ml]",
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        except Exception as exc:  # noqa: BLE001
            return json.dumps(
                {
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                    "module": "relation_induction",
                    "model": self.MODEL_NAME,
                    "relation_ontology": list(self.RELATION_ONTOLOGY),
                },
                ensure_ascii=True,
                sort_keys=True,
            )

    def _parse_input(self, text: str) -> dict[str, Any]:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Input must be a JSON object with 'text' and 'entities'")
        if "text" not in data:
            raise ValueError("Input must contain 'text' field")
        entities = data.get("entities", [])
        if not isinstance(entities, list):
            raise ValueError("'entities' must be a list")
        return data

    def _extract_relations(self, data: dict[str, Any]) -> list[RelationCandidate]:
        self._ensure_model()
        text = data["text"]
        entities = data.get("entities", [])

        if not text.strip() or not entities:
            return []

        # Process with spaCy
        doc = self._nlp(text)  # type: ignore[misc]

        # Build entity index by position (with overlap tolerance)
        entity_map: dict[tuple[int, int], dict[str, Any]] = {}
        for ent in entities:
            start = ent.get("start", -1)
            end = ent.get("end", -1)
            if start >= 0 and end > start:
                entity_map[(start, end)] = ent

        def find_overlapping_entity(start: int, end: int) -> dict[str, Any] | None:
            """Find entity that overlaps with given span."""
            for (e_start, e_end), entity in entity_map.items():
                # Check if ranges overlap
                if not (end <= e_start or start >= e_end):
                    return entity
            return None

        candidates: list[RelationCandidate] = []

        # Extract predicate-argument structures from dependency parse
        for token in doc:
            if token.pos_ == "VERB":
                predicate = token.lemma_
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                objects = [child for child in token.children if child.dep_ in ("dobj", "pobj", "attr")]

                for subj in subjects:
                    for obj in objects:
                        subj_start, subj_end = subj.idx, subj.idx + len(subj.text)
                        obj_start, obj_end = obj.idx, obj.idx + len(obj.text)

                        # Try to find overlapping entities
                        subj_entity = find_overlapping_entity(subj_start, subj_end)
                        obj_entity = find_overlapping_entity(obj_start, obj_end)

                        if subj_entity and obj_entity:
                            relation_type = self._map_predicate_to_relation(predicate)
                            confidence = self._compute_confidence(token)

                            provenance = {
                                "model": self.MODEL_NAME,
                                "relation_ontology": list(self.RELATION_ONTOLOGY),
                                "strategy": "spacy_dependency_parse",
                                "predicate_pos": token.pos_,
                                "dependency_pattern": f"{subj.dep_}-{token.dep_}-{obj.dep_}",
                            }

                            candidates.append(
                                RelationCandidate(
                                    relation_type=relation_type,
                                    subject_span=subj_entity["text"],
                                    subject_start=subj_entity["start"],
                                    subject_end=subj_entity["end"],
                                    object_span=obj_entity["text"],
                                    object_start=obj_entity["start"],
                                    object_end=obj_entity["end"],
                                    predicate=predicate,
                                    confidence=confidence,
                                    provenance=provenance,
                                )
                            )

        return candidates

    def _map_predicate_to_relation(self, predicate: str) -> str:
        """Map verb lemma to closed relation ontology."""
        # Simple deterministic mapping
        cause_verbs = {"cause", "trigger", "induce", "lead", "produce", "generate"}
        part_verbs = {"contain", "include", "comprise", "consist"}
        use_verbs = {"use", "utilize", "employ", "apply"}
        depend_verbs = {"require", "need", "depend", "rely"}
        contradict_verbs = {"contradict", "oppose", "conflict", "disagree"}

        if predicate in cause_verbs:
            return "CAUSES"
        elif predicate in part_verbs:
            return "PART_OF"
        elif predicate in use_verbs:
            return "USES"
        elif predicate in depend_verbs:
            return "DEPENDS_ON"
        elif predicate in contradict_verbs:
            return "CONTRADICTS"
        else:
            return "RELATED_TO"

    def _compute_confidence(self, token: Any) -> float:
        """Compute confidence based on syntactic features."""
        # Deterministic confidence based on dependency depth and verb properties
        base_confidence = 0.7
        if hasattr(token, "head") and token.head.pos_ == "ROOT":
            base_confidence += 0.15
        if len(list(token.children)) > 2:
            base_confidence += 0.1
        return min(1.0, base_confidence)

    def _ensure_model(self) -> None:
        if self._nlp is not None:
            return

        try:
            import importlib
            spacy = importlib.import_module("spacy")  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover
            raise _MissingOptionalDependency({"spacy"}) from exc

        try:
            self._nlp = spacy.load(self.MODEL_NAME)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover
            # Model not downloaded
            raise RuntimeError(
                f"spaCy model '{self.MODEL_NAME}' not found. "
                f"Install with: python -m spacy download {self.MODEL_NAME}"
            ) from exc
