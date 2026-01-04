"""Curriculum sorting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset

THINKING_OPEN = "<thinking>"
THINKING_CLOSE = "</thinking>"


@dataclass
class CurriculumSorter:
    """Sort datasets from easiest to hardest based on reasoning length."""

    score_column: str = "complexity_score"

    def sort(self, dataset: Dataset) -> Dataset:
        scored = dataset.map(self._add_score)
        return scored.sort(self.score_column)

    def _add_score(self, example: dict[str, Any]) -> dict[str, Any]:
        text = str(example.get("text", "")).strip()
        score = len(text)
        if THINKING_OPEN in text and THINKING_CLOSE in text:
            thinking = text.split(THINKING_OPEN, 1)[1].split(THINKING_CLOSE, 1)[0]
            score = len([tok for tok in thinking.split() if tok])
        return {self.score_column: score}
