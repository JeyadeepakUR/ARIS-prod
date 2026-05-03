"""Deterministic mock LLM provider for tests and environments without API keys."""

from __future__ import annotations

import hashlib
import json


class MockProvider:
    """Returns deterministic, content-aware mock responses for testing."""

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        prompt_lower = prompt.lower()

        # ── New LangGraph agent prompts (checked first, most specific) ─────────
        if "scientific concept extractor" in prompt_lower or (
            "extract" in prompt_lower and "concepts" in prompt_lower and "label" in prompt_lower
        ):
            return self._concept_list_response(prompt)

        if "research bridge analyst" in prompt_lower or (
            "meaningful research bridge" in prompt_lower
        ):
            return self._bridge_validation_response(prompt)

        if "multi-hop" in prompt_lower or "indirect bridge" in prompt_lower:
            return self._multihop_response(prompt)

        if "contradictory claims" in prompt_lower or (
            "contradicts" in prompt_lower and "excerpt" in prompt_lower
        ):
            return self._contradiction_response(prompt)

        if "research gap" in prompt_lower and "investigation_priority" in prompt_lower:
            return self._gap_response(prompt)

        # ── Legacy / other prompts ─────────────────────────────────────────────
        if "falsifiable" in prompt_lower or (
            "hypothesis" in prompt_lower and "statement" in prompt_lower
        ):
            return self._hypothesis_response(prompt)
        if "named research concepts" in prompt_lower or (
            "knowledge graph builder" in prompt_lower and "domains" in prompt_lower
        ):
            return self._concept_extraction_response(prompt)
        if "bridge concept" in prompt_lower or "bridge_concept" in prompt_lower:
            return self._bridge_response(prompt)
        if "classify" in prompt_lower and ("tier" in prompt_lower or "domain" in prompt_lower):
            return self._classify_response(prompt)
        if "hypothesis" in prompt_lower:
            return self._hypothesis_response(prompt)
        if "domain" in prompt_lower and "extract" in prompt_lower:
            return self._domain_response(prompt)
        if "evidence" in prompt_lower or "justify" in prompt_lower or "link" in prompt_lower:
            return self._evidence_response(prompt)

        digest = hashlib.sha256(prompt[:200].encode()).hexdigest()[:8]
        return f"[mock:{digest}] Deterministic response for: {prompt[:80]}"

    def complete_json(self, prompt: str, *, max_tokens: int = 512) -> dict:
        text = self.complete(prompt, max_tokens=max_tokens)
        import re
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"raw": text}

    # ── New agent response helpers ─────────────────────────────────────────────

    def _concept_list_response(self, prompt: str) -> str:
        """Return concept list in the format expected by concept_extractor_node."""
        prompt_lower = prompt.lower()
        concepts = []

        if any(w in prompt_lower for w in ["machine learning", "neural", "deep learning", "training"]):
            concepts += [
                {"label": "deep neural network", "domain": "Machine Learning", "confidence": 0.9},
                {"label": "transfer learning", "domain": "Machine Learning", "confidence": 0.85},
            ]
        if any(w in prompt_lower for w in ["security", "attack", "intrusion", "threat", "malware"]):
            concepts += [
                {"label": "intrusion detection system", "domain": "Cybersecurity", "confidence": 0.88},
                {"label": "anomaly detection", "domain": "Cybersecurity", "confidence": 0.82},
            ]
        if any(w in prompt_lower for w in ["image", "vision", "object", "detection"]):
            concepts += [
                {"label": "object detection pipeline", "domain": "Computer Vision", "confidence": 0.86},
            ]
        if any(w in prompt_lower for w in ["language", "nlp", "text", "bert"]):
            concepts += [
                {"label": "pre-trained language model", "domain": "Natural Language Processing", "confidence": 0.85},
            ]
        if any(w in prompt_lower for w in ["clinical", "patient", "diagnosis", "health", "medical"]):
            concepts += [
                {"label": "clinical decision support", "domain": "Healthcare", "confidence": 0.87},
            ]

        # Always return at least 2 concepts
        if len(concepts) < 2:
            concepts = [
                {"label": "gradient-based optimization", "domain": "Machine Learning", "confidence": 0.88},
                {"label": "representation learning", "domain": "Machine Learning", "confidence": 0.80},
            ]

        return json.dumps({"concepts": concepts[:5]})

    def _bridge_validation_response(self, prompt: str) -> str:
        """Validate cross-domain bridge — always valid in mock for testability."""
        return json.dumps({
            "valid": True,
            "bridge_concept": "shared optimization objective",
            "confidence": 0.78,
            "explanation": (
                "Both concepts rely on gradient-based optimization, enabling direct "
                "transfer of regularisation techniques across domains."
            ),
        })

    def _multihop_response(self, prompt: str) -> str:
        return json.dumps({
            "bridge_concept": "iterative feature refinement pathway",
            "confidence": 0.65,
            "explanation": (
                "The indirect chain propagates a shared learning paradigm through "
                "an intermediate representation step."
            ),
        })

    def _contradiction_response(self, prompt: str) -> str:
        """Return non-contradicting by default to keep tests deterministic."""
        return json.dumps({
            "contradicts": False,
            "contradiction_type": None,
            "severity": 0.0,
            "reasoning": "The excerpts discuss related but non-contradictory aspects.",
            "claim_a_summary": "Excerpt A describes a method or result.",
            "claim_b_summary": "Excerpt B extends or complements excerpt A.",
        })

    def _gap_response(self, prompt: str) -> str:
        return json.dumps({
            "gap_description": (
                "No direct study of the relationship between these two concepts has been "
                "published despite their frequent co-occurrence via shared intermediaries. "
                "This gap limits cross-domain synthesis."
            ),
            "investigation_priority": 0.72,
            "rationale": (
                "Researchers in both fields should collaborate on a controlled study "
                "that explicitly tests the transfer mechanism."
            ),
        })

    # ── Legacy response helpers ────────────────────────────────────────────────

    def _bridge_response(self, prompt: str) -> str:
        return json.dumps({
            "bridge_concept": "shared optimization objective",
            "description": (
                "Both domains converge on gradient-based optimization of a loss function, "
                "enabling direct transfer of regularization and convergence techniques."
            ),
            "novelty": 0.72,
            "transfer_mechanism": "mathematical_formalism",
        })

    def _classify_response(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ["domain", "field", "area", "discipline"]):
            tier = 1
        elif any(w in prompt_lower for w in ["method", "technique", "approach", "algorithm", "model"]):
            tier = 2
        else:
            tier = 3

        cluster = "General Research"
        for domain in ["machine learning", "security", "healthcare", "vision", "nlp", "blockchain"]:
            if domain in prompt_lower:
                cluster = domain.title()
                break

        return json.dumps({"tier": tier, "cluster_id": cluster, "low_value": False})

    def _hypothesis_response(self, prompt: str) -> str:
        return json.dumps({
            "statement": (
                "We hypothesize that the observed performance gap between the source and target "
                "methods narrows when the shared intermediate representation is explicitly "
                "aligned during training."
            ),
            "null_hypothesis": (
                "Explicit alignment of intermediate representations does not reduce "
                "the performance gap."
            ),
            "methodology_hint": (
                "Implement a contrastive alignment loss on the intermediate layers and "
                "evaluate on held-out benchmarks from both domains."
            ),
            "evidence_basis": "Bridge edge confidence and shared concept co-occurrence.",
            "hypothesis_type": "causal",
            "testability_score": 0.82,
            "confidence": 0.75,
        })

    def _concept_extraction_response(self, prompt: str) -> str:
        """Legacy format: return domain-grouped concepts for old pipeline."""
        prompt_lower = prompt.lower()
        domains = []
        if any(w in prompt_lower for w in ["image", "vision", "caption", "object"]):
            domains.append({
                "name": "computer_vision",
                "concepts": ["image captioning", "visual attention mechanism", "object detection pipeline"],
            })
        if any(w in prompt_lower for w in ["language", "text", "nlp", "bert", "transformer"]):
            domains.append({
                "name": "natural_language_processing",
                "concepts": ["pre-trained language model", "attention-based encoding", "text generation"],
            })
        if any(w in prompt_lower for w in ["deep", "neural", "training", "gradient", "loss"]):
            domains.append({
                "name": "machine_learning",
                "concepts": ["deep neural network", "transfer learning", "contrastive learning"],
            })
        if any(w in prompt_lower for w in ["security", "attack", "threat", "malware", "intrusion"]):
            domains.append({
                "name": "cybersecurity",
                "concepts": ["adversarial attack detection", "intrusion detection system", "threat modelling"],
            })
        if len(domains) < 2:
            domains = [
                {"name": "machine_learning", "concepts": ["deep neural network", "transfer learning"]},
                {"name": "computer_vision", "concepts": ["image captioning", "visual attention mechanism"]},
            ]
        return json.dumps({"domains": domains[:4]})

    def _domain_response(self, prompt: str) -> str:
        return json.dumps({
            "domains": ["machine learning", "optimization"],
            "primary_domain": "machine learning",
        })

    def _evidence_response(self, prompt: str) -> str:
        return (
            "The two documents share a common methodological foundation: both apply "
            "iterative optimization over a parameterized function space with gradient-based updates. "
            "Document A demonstrates performance gains of 12% on benchmark X, while Document B "
            "replicates this pattern in a different modality, suggesting the mechanism generalizes. "
            "The shared evidence supports a 'supports' relationship with high confidence."
        )
