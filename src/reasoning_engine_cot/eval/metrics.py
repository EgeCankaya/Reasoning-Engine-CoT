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
