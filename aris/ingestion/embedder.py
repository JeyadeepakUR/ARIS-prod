"""
Embedding generator.

Strategy (in order):
  1. Ollama  /api/embed  — local, free, nomic-embed-text (768-dim)
  2. sentence-transformers all-mpnet-base-v2 — pure Python, no server, 768-dim

Both produce 768-dimensional vectors compatible with the pgvector schema.

The fallback model is lazy-loaded on first use and cached for the process
lifetime so repeated calls are fast.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Expected output dimension from nomic-embed-text and all-mpnet-base-v2
EMBED_DIM = 768

# fastembed model used when Ollama is unavailable.
# nomic-embed-text-v1.5 is 768-dim and the same model family as Ollama's nomic-embed-text,
# so vectors are compatible with existing DB rows.
_FALLBACK_FASTEMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"


class Embedder:
    """Generate 768-dimensional embeddings for text chunks."""

    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        fallback_to_local: bool = True,
        timeout: float = 30.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._fallback = fallback_to_local
        self._timeout = timeout
        self._fastembed_model = None   # lazy-loaded fastembed TextEmbedding instance
        self._ollama_available: bool | None = None  # cached after first probe

    def embed(self, text: str) -> list[float]:
        """Return a 768-dimensional embedding for the given text."""
        results = self.embed_batch([text])
        return results[0]

    # Maximum chunks per Ollama API call — 5 keeps each call under ~15s on CPU Ollama.
    OLLAMA_BATCH_SIZE = 5

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, batching Ollama calls to avoid timeouts.

        Probes Ollama once per Embedder instance; if unreachable, skips it for
        all subsequent batches so we don't waste time on repeated connection
        failures.
        """
        if not texts:
            return []

        # Probe once and cache the result for the lifetime of this instance.
        if self._ollama_available is None:
            self._ollama_available = self.is_available()
            if not self._ollama_available:
                logger.info("Ollama not reachable — using fastembed for all embeddings")

        if not self._ollama_available or not self._fallback:
            # Go straight to fastembed; no per-batch Ollama attempts.
            results: list[list[float]] = []
            for i in range(0, len(texts), self.OLLAMA_BATCH_SIZE):
                results.extend(self._st_embed_batch(texts[i : i + self.OLLAMA_BATCH_SIZE]))
            return results

        results = []
        for i in range(0, len(texts), self.OLLAMA_BATCH_SIZE):
            batch = texts[i : i + self.OLLAMA_BATCH_SIZE]
            try:
                results.extend(self._ollama_embed_batch(batch))
            except Exception as exc:
                logger.warning(
                    "Ollama embed failed for batch %d-%d (%s) — falling back to fastembed",
                    i, i + len(batch) - 1, exc,
                )
                self._ollama_available = False  # stop trying for remaining batches
                results.extend(self._st_embed_batch(batch))
        return results

    def is_available(self) -> bool:
        """Quick liveness check — returns True if Ollama is reachable."""
        try:
            import httpx
            httpx.get(f"{self._base_url}/api/tags", timeout=3.0)
            return True
        except Exception:
            return False

    # ── Ollama ───────────────────────────────────────────────────────────────

    def _ollama_embed_batch(self, texts: list[str]) -> list[list[float]]:
        import httpx

        # /api/embed supports batched input (Ollama ≥ 0.1.31)
        resp = httpx.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": texts},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # Response shape: {"embeddings": [[...], [...]]}
        embeddings: list[list[float]] = data.get("embeddings", [])
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )
        return embeddings

    # ── fastembed fallback (ONNX — ~10x faster than sentence-transformers on CPU) ─

    def _st_embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = self._load_fastembed_model()
        # fastembed.embed() returns a generator of numpy arrays
        return [v.tolist() for v in model.embed(texts)]

    def _load_fastembed_model(self):
        if self._fastembed_model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ImportError(
                    "fastembed is required as embedding fallback. "
                    "Run: pip install fastembed"
                ) from exc
            logger.info("Loading fastembed model %s (first use)", _FALLBACK_FASTEMBED_MODEL)
            self._fastembed_model = TextEmbedding(_FALLBACK_FASTEMBED_MODEL)
        return self._fastembed_model
