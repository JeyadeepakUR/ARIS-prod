"""SRL-Enhanced Relation Induction (Module 12).

Semantic relation extraction strictly as hypothesis generation.
- No graph edge creation
- Closed relation ontology
- Deterministic inference
- Full metadata logging
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

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
    """Context-aware semantic relation extraction with discourse patterns (Module 12).

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
        "ACHIEVES",
        "OUTPERFORMS",
        "IMPROVES",
        "SUPPORTS",
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
        text = data["text"]
        entities = data.get("entities", [])

        if not text.strip() or not entities:
            return []

        doc = self._maybe_parse(text)
        sentences = list(self._iter_sentences(text, doc))

        # Build entity index by position (with overlap tolerance)
        entity_map: dict[tuple[int, int], dict[str, Any]] = {}
        for ent in entities:
            start = ent.get("start", -1)
            end = ent.get("end", -1)
            if start >= 0 and end > start:
                entity_map[(start, end)] = ent

        candidates: list[RelationCandidate] = []

        # Pattern + discourse extraction (novel component)
        candidates.extend(self._extract_pattern_relations(sentences, entity_map))

        # Dependency-based fallback when spaCy is available
        if doc is not None:
            candidates.extend(self._extract_dependency_relations(doc, entity_map))

        return self._deduplicate(candidates)

    def _extract_dependency_relations(self, doc: Any, entity_map: dict[tuple[int, int], dict[str, Any]]) -> list[RelationCandidate]:
        def find_overlapping_entity(start: int, end: int) -> dict[str, Any] | None:
            for (e_start, e_end), entity in entity_map.items():
                if not (end <= e_start or start >= e_end):
                    return entity
            return None

        candidates: list[RelationCandidate] = []

        for token in doc:
            if getattr(token, "pos_", "") == "VERB":
                predicate = getattr(token, "lemma_", getattr(token, "text", "")) or ""
                subjects = [child for child in token.children if getattr(child, "dep_", "") in ("nsubj", "nsubjpass")]
                objects = [child for child in token.children if getattr(child, "dep_", "") in ("dobj", "pobj", "attr")]

                for subj in subjects:
                    for obj in objects:
                        subj_start, subj_end = subj.idx, subj.idx + len(subj.text)
                        obj_start, obj_end = obj.idx, obj.idx + len(obj.text)

                        subj_entity = find_overlapping_entity(subj_start, subj_end)
                        obj_entity = find_overlapping_entity(obj_start, obj_end)

                        if subj_entity and obj_entity:
                            relation_type = self._map_predicate_to_relation(predicate)
                            confidence = self._compute_dependency_confidence(token)

                            provenance = {
                                "model": self.MODEL_NAME,
                                "relation_ontology": list(self.RELATION_ONTOLOGY),
                                "strategy": "spacy_dependency_parse",
                                "predicate_pos": getattr(token, "pos_", ""),
                                "dependency_pattern": f"{getattr(subj, 'dep_', '')}-{getattr(token, 'dep_', '')}-{getattr(obj, 'dep_', '')}",
                            }

                            candidates.append(
                                RelationCandidate(
                                    relation_type=relation_type,
                                    subject_span=self._entity_span(subj_entity),
                                    subject_start=subj_entity.get("start", -1),
                                    subject_end=subj_entity.get("end", -1),
                                    object_span=self._entity_span(obj_entity),
                                    object_start=obj_entity.get("start", -1),
                                    object_end=obj_entity.get("end", -1),
                                    predicate=predicate,
                                    confidence=confidence,
                                    provenance=provenance,
                                )
                            )

        return candidates

    def _extract_pattern_relations(
        self,
        sentences: list[dict[str, Any]],
        entity_map: dict[tuple[int, int], dict[str, Any]],
    ) -> list[RelationCandidate]:
        candidates: list[RelationCandidate] = []

        method_types = {"METHOD", "MODEL", "ALGORITHM", "APPROACH"}
        metric_types = {"METRIC", "SCORE", "ACCURACY", "BLEU"}
        dataset_types = {"DATASET", "CORPUS", "BENCHMARK"}

        for sentence_id, sent in enumerate(sentences):
            sent_text = sent["text"]
            sent_start = sent["start"]
            sent_end = sent["end"]
            sent_lower = sent_text.lower()

            discourse_cues = [cue for cue in ("result", "experiment", "we show", "significant", "compared") if cue in sent_lower]

            entities_in_sentence: list[dict[str, Any]] = []
            for (s, e), ent in entity_map.items():
                if s >= sent_start and e <= sent_end:
                    entities_in_sentence.append(ent)

            if len(entities_in_sentence) < 2:
                continue

            method_entities = [ent for ent in entities_in_sentence if ent.get("type", "").upper() in method_types]
            metric_entities = [ent for ent in entities_in_sentence if ent.get("type", "").upper() in metric_types]
            dataset_entities = [ent for ent in entities_in_sentence if ent.get("type", "").upper() in dataset_types]

            # ACHIEVES: method + metric with achievement verbs
            if method_entities and metric_entities and re.search(r"achiev|reach|score|obtain", sent_lower):
                for method_ent in method_entities:
                    for metric_ent in metric_entities:
                        candidates.append(
                            self._build_candidate(
                                relation_type="ACHIEVES",
                                subject_ent=method_ent,
                                object_ent=metric_ent,
                                predicate="achieve",
                                cues=discourse_cues,
                                strong_pattern=True,
                                type_alignment=True,
                                sentence_id=sentence_id,
                                pattern="pattern_achieves",
                            )
                        )

            # OUTPERFORMS: method vs method comparative
            if len(method_entities) >= 2 and re.search(r"outperform|better than|beat|surpass|exceed", sent_lower):
                primary = method_entities[0]
                baseline = method_entities[1]
                candidates.append(
                    self._build_candidate(
                        relation_type="OUTPERFORMS",
                        subject_ent=primary,
                        object_ent=baseline,
                        predicate="outperform",
                        cues=discourse_cues,
                        strong_pattern=True,
                        type_alignment=True,
                        sentence_id=sentence_id,
                        pattern="pattern_outperforms",
                    )
                )

            # IMPROVES: method + metric improvement language
            if method_entities and metric_entities and re.search(r"improv|increase|reduce error", sent_lower):
                for method_ent in method_entities:
                    for metric_ent in metric_entities:
                        candidates.append(
                            self._build_candidate(
                                relation_type="IMPROVES",
                                subject_ent=method_ent,
                                object_ent=metric_ent,
                                predicate="improve",
                                cues=discourse_cues,
                                strong_pattern=False,
                                type_alignment=True,
                                sentence_id=sentence_id,
                                pattern="pattern_improves",
                            )
                        )

            # USES: method uses dataset/material
            if method_entities and dataset_entities and re.search(r"use|based on|built on|leverag", sent_lower):
                for method_ent in method_entities:
                    for data_ent in dataset_entities:
                        candidates.append(
                            self._build_candidate(
                                relation_type="USES",
                                subject_ent=method_ent,
                                object_ent=data_ent,
                                predicate="use",
                                cues=discourse_cues,
                                strong_pattern=False,
                                type_alignment=True,
                                sentence_id=sentence_id,
                                pattern="pattern_uses",
                            )
                        )

        return candidates

    def _build_candidate(
        self,
        *,
        relation_type: str,
        subject_ent: dict[str, Any],
        object_ent: dict[str, Any],
        predicate: str,
        cues: list[str],
        strong_pattern: bool,
        type_alignment: bool,
        sentence_id: int,
        pattern: str,
    ) -> RelationCandidate:
        confidence = self._pattern_confidence(cues, strong_pattern, type_alignment)
        provenance = {
            "strategy": "pattern_discourse",
            "relation_ontology": list(self.RELATION_ONTOLOGY),
            "pattern": pattern,
            "discourse_cues": cues,
            "sentence_id": sentence_id,
        }
        return RelationCandidate(
            relation_type=relation_type,
            subject_span=self._entity_span(subject_ent),
            subject_start=subject_ent.get("start", -1),
            subject_end=subject_ent.get("end", -1),
            object_span=self._entity_span(object_ent),
            object_start=object_ent.get("start", -1),
            object_end=object_ent.get("end", -1),
            predicate=predicate,
            confidence=confidence,
            provenance=provenance,
        )

    def _pattern_confidence(self, cues: list[str], strong_pattern: bool, type_alignment: bool) -> float:
        base = 0.65
        if strong_pattern:
            base += 0.15
        if type_alignment:
            base += 0.1
        base += min(0.15, 0.05 * len(cues))
        return min(1.0, base)

    def _compute_dependency_confidence(self, token: Any) -> float:
        base_confidence = 0.7
        if hasattr(token, "head") and getattr(token.head, "pos_", "") == "ROOT":
            base_confidence += 0.15
        if len(list(getattr(token, "children", []))) > 2:
            base_confidence += 0.1
        return min(1.0, base_confidence)

    def _map_predicate_to_relation(self, predicate: str) -> str:
        cause_verbs = {"cause", "trigger", "induce", "lead", "produce", "generate"}
        part_verbs = {"contain", "include", "comprise", "consist"}
        use_verbs = {"use", "utilize", "employ", "apply"}
        depend_verbs = {"require", "need", "depend", "rely"}
        contradict_verbs = {"contradict", "oppose", "conflict", "disagree"}
        achieve_verbs = {"achieve", "reach", "score", "obtain"}
        outperform_verbs = {"outperform", "surpass", "exceed", "beat"}
        improve_verbs = {"improve", "increase", "reduce"}
        support_verbs = {"support", "validate", "confirm"}

        if predicate in cause_verbs:
            return "CAUSES"
        if predicate in part_verbs:
            return "PART_OF"
        if predicate in use_verbs:
            return "USES"
        if predicate in depend_verbs:
            return "DEPENDS_ON"
        if predicate in contradict_verbs:
            return "CONTRADICTS"
        if predicate in achieve_verbs:
            return "ACHIEVES"
        if predicate in outperform_verbs:
            return "OUTPERFORMS"
        if predicate in improve_verbs:
            return "IMPROVES"
        if predicate in support_verbs:
            return "SUPPORTS"
        return "RELATED_TO"

    def _maybe_parse(self, text: str) -> Any | None:
        try:
            self._ensure_model()
        except _MissingOptionalDependency:
            raise
        except Exception:
            return None

        if self._nlp is None:
            return None
        return self._nlp(text)  # type: ignore[misc]

    def _iter_sentences(self, text: str, doc: Any | None) -> Iterable[dict[str, Any]]:
        if doc is not None and hasattr(doc, "sents"):
            for sent in doc.sents:
                start = getattr(sent, "start_char", 0)
                end = getattr(sent, "end_char", start + len(getattr(sent, "text", "")))
                yield {"text": getattr(sent, "text", ""), "start": start, "end": end}
            return

        segments = re.split(r"(?<=[.!?])\s+", text)
        offset = 0
        for segment in segments:
            seg = segment.strip()
            if not seg:
                offset += len(segment) + 1
                continue
            start = text.find(seg, offset)
            end = start + len(seg)
            yield {"text": seg, "start": start, "end": end}
            offset = end

    def _deduplicate(self, candidates: list[RelationCandidate]) -> list[RelationCandidate]:
        seen: set[tuple[str, int, int, str]] = set()
        deduped: list[RelationCandidate] = []
        for cand in candidates:
            key = (cand.relation_type, cand.subject_start, cand.object_start, cand.predicate)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cand)
        return deduped

    def _entity_span(self, entity: dict[str, Any]) -> str:
        return entity.get("span") or entity.get("text") or ""

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
        except Exception:
            self._nlp = None
