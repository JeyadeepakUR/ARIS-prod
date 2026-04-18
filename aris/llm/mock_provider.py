"""Deterministic mock LLM provider for tests and environments without API keys."""

from __future__ import annotations

import hashlib
import json


class MockProvider:
    """Returns deterministic, content-aware mock responses for testing."""

    def complete(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.2) -> str:
        prompt_lower = prompt.lower()

        # Order matters: more specific checks first.
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
                "The observed performance gap between the source and target methods narrows "
                "when the shared intermediate representation is explicitly aligned during training."
            ),
            "null_hypothesis": (
                "Explicit alignment of intermediate representations does not reduce the performance gap."
            ),
            "methodology_hint": (
                "Implement a contrastive alignment loss on the intermediate layers; "
                "evaluate on held-out benchmarks from both domains."
            ),
            "evidence_basis": "Bridge edge confidence and shared concept co-occurrence in both documents.",
            "hypothesis_type": "causal",
        })

    def _concept_extraction_response(self, prompt: str) -> str:
        """Return mock named concepts for concept-extraction prompts."""
        prompt_lower = prompt.lower()
        # Pick domains based on keywords in the excerpt
        domains = []
        if any(w in prompt_lower for w in ["image", "vision", "caption", "object"]):
            domains.append({
                "name": "computer_vision",
                "concepts": ["image captioning", "visual attention mechanism", "convolutional feature extraction", "object detection pipeline", "vision transformer"],
            })
        if any(w in prompt_lower for w in ["language", "text", "nlp", "bert", "gpt", "transformer"]):
            domains.append({
                "name": "natural_language_processing",
                "concepts": ["pre-trained language model", "attention-based encoding", "sequence-to-sequence learning", "text generation", "semantic representation"],
            })
        if any(w in prompt_lower for w in ["deep", "neural", "training", "gradient", "loss"]):
            domains.append({
                "name": "machine_learning",
                "concepts": ["deep neural network", "transfer learning", "contrastive learning", "fine-tuning strategy", "self-supervised pretraining"],
            })
        if any(w in prompt_lower for w in ["security", "attack", "threat", "malware", "intrusion"]):
            domains.append({
                "name": "cybersecurity",
                "concepts": ["adversarial attack detection", "intrusion detection system", "threat modelling", "anomaly-based detection", "federated security"],
            })
        if any(w in prompt_lower for w in ["blockchain", "consensus", "ledger", "smart contract"]):
            domains.append({
                "name": "blockchain",
                "concepts": ["proof-of-stake consensus", "smart contract execution", "decentralised identity", "on-chain governance", "zero-knowledge proof"],
            })
        if any(w in prompt_lower for w in ["medical", "clinical", "patient", "diagnosis", "health"]):
            domains.append({
                "name": "healthcare",
                "concepts": ["clinical decision support", "medical image analysis", "patient outcome prediction", "electronic health record", "biomarker detection"],
            })
        # Ensure at least two domains
        if len(domains) < 2:
            domains = [
                {"name": "machine_learning", "concepts": ["deep neural network", "transfer learning", "contrastive learning", "self-supervised pretraining", "model fine-tuning"]},
                {"name": "computer_vision", "concepts": ["image captioning", "visual attention mechanism", "object detection", "feature extraction", "visual grounding"]},
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
