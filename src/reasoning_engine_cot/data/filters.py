"""Quality filtering for training datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset

THINKING_OPEN = "<thinking>"
THINKING_CLOSE = "</thinking>"


@dataclass
class QualityFilter:
    """Filter out low-quality or malformed samples before training."""

    min_text_length: int = 100
    min_reasoning_words: int = 10
    require_tags: bool = True
    max_text_length: int = 8000

    def filter(self, dataset: Dataset) -> Dataset:
        return dataset.filter(self._is_valid)

    def _is_valid(self, example: dict[str, Any]) -> bool:
        text = str(example.get("text", "")).strip()
        if not text:
            return False
        if len(text) < self.min_text_length:
            return False
        if self.max_text_length and len(text) > self.max_text_length:
            return False
        if self.require_tags and (THINKING_OPEN not in text or THINKING_CLOSE not in text):
            return False

        thinking = self._extract_thinking(text)
        if thinking is not None and self._word_count(thinking) < self.min_reasoning_words:
            return False
        return True

    def _extract_thinking(self, text: str) -> str | None:
        if THINKING_OPEN in text and THINKING_CLOSE in text:
            return text.split(THINKING_OPEN, 1)[1].split(THINKING_CLOSE, 1)[0].strip()
        return None

    @staticmethod
    def _word_count(text: str) -> int:
        return len([tok for tok in text.split() if tok])
