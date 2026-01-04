from __future__ import annotations

from contextlib import suppress
from typing import Any

import torch


def strict_adherence(text: str) -> tuple[bool, bool]:
    """Returns (thinking_ok, answer_ok) requiring both open/close tags."""
    return ("<thinking>" in text and "</thinking>" in text, "<answer>" in text and "</answer>" in text)


def recovered_adherence(text: str) -> tuple[bool, bool]:
    """Lenient adherence: both tags mentioned anywhere, even if not closed."""
    return ("<thinking>" in text, "<answer>" in text)


def tokens_stats(tokenizer: Any, text: str, latency_ms: float) -> tuple[int, float, float]:
    tokens = len(tokenizer.encode(text))
    ms_per_token = latency_ms / max(tokens, 1)
    tokens_per_sec = 1000.0 / ms_per_token if ms_per_token > 0 else 0.0
    return tokens, ms_per_token, tokens_per_sec


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        with suppress(Exception):
            torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> float | None:
    if torch.cuda.is_available():
        try:
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            return None
    return None


class SemanticMatcher:
    """Embedding-based semantic similarity scorer."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = None
        with suppress(Exception):
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)

    def similarity(self, predicted: str, expected: str) -> float:
        if self.model is None:
            raise RuntimeError("sentence-transformers model not available")
        embeddings = self.model.encode([predicted, expected], convert_to_numpy=True, normalize_embeddings=True)
        pred_vec, exp_vec = embeddings
        return float(pred_vec @ exp_vec)


class ReasoningCoherenceScorer:
    """Heuristic coherence scorer: more structured reasoning that contains the answer scores higher."""

    def score(self, thinking: str | None, answer: str | None) -> float:
        if not thinking or not answer:
            return 0.0
        thinking = thinking.strip()
        answer = answer.strip()
        if not thinking or not answer:
            return 0.0

        steps = self._count_steps(thinking)
        contains_answer = answer.lower() in thinking.lower()
        length_bonus = min(len(thinking.split()) / 200.0, 0.3)
        step_bonus = min(steps / 8.0, 0.4)
        answer_bonus = 0.3 if contains_answer else 0.0
        return float(min(1.0, length_bonus + step_bonus + answer_bonus))

    @staticmethod
    def _count_steps(thinking: str) -> int:
        markers = ["\n- ", "\n* ", "1.", "2.", "3."]
        for marker in markers:
            if marker in thinking:
                return max(len([chunk for chunk in thinking.split(marker) if chunk.strip()]), 1)
        return max(thinking.count("\n"), 1)
