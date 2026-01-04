"""Build preference datasets for DPO training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset


@dataclass
class PreferenceBuilder:
    prompt_column: str = "prompt"
    text_column: str = "text"
    score_column: str = "quality_score"

    def build(self, dataset: Dataset) -> Dataset:
        """Construct a preference dataset with prompt/chosen/rejected fields."""
        records = dataset.to_list()
        scored = [
            (
                float(item.get(self.score_column, 0.0)),
                str(item.get(self.prompt_column, "")),
                str(item.get(self.text_column, "")),
            )
            for item in records
        ]
        if len(scored) < 2:
            raise ValueError("Need at least two samples to build preference pairs.")

        scored.sort(key=lambda tup: tup[0], reverse=True)
        pairs: list[dict[str, Any]] = []
        i, j = 0, len(scored) - 1
        while i < j:
            hi, prompt_hi, text_hi = scored[i]
            lo, prompt_lo, text_lo = scored[j]
            prompt = prompt_hi or prompt_lo
            pairs.append({"prompt": prompt, "chosen": text_hi, "rejected": text_lo, "score_delta": hi - lo})
            i += 1
            j -= 1

        return Dataset.from_list(pairs)
