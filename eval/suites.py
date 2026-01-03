from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset


@dataclass
class Example:
    id: str
    question: str
    answer: str


def load_riddles(path: Path) -> list[Example]:
    items: list[Example] = []
    with path.open() as f:
        for idx, line in enumerate(f):
            import json  # lazy import

            obj = json.loads(line)
            items.append(Example(id=obj.get("id", f"riddle_{idx}"), question=obj["question"], answer=obj["answer"]))
    return items


def load_gsm8k_lite(limit: int = 100, split: str = "test") -> list[Example]:
    # gsm8k is public; using "main" config
    ds = load_dataset("gsm8k", "main", split=split)
    rows = ds.select(range(min(limit, len(ds))))
    examples: list[Example] = []
    for i, row in enumerate(rows):
        question = row["question"]
        # gsm8k answers include reasoning steps; keep final line after '#### ' as canonical answer
        ans = row["answer"]
        final = ans.split("####")[-1].strip() if "####" in ans else ans.strip()
        examples.append(Example(id=f"gsm8k_{i}", question=question, answer=final))
    return examples


def load_suite(name: str, riddles_path: Path = Path("eval/questions.jsonl"), gsm8k_limit: int = 100) -> list[Example]:
    name = name.lower()
    if name == "riddles":
        return load_riddles(riddles_path)
    if name in ("gsm8k", "gsm8k-lite", "gsm8k_lite"):
        return load_gsm8k_lite(limit=gsm8k_limit)
    if name == "all":
        combined: list[Example] = []
        combined.extend(load_riddles(riddles_path))
        combined.extend(load_gsm8k_lite(limit=gsm8k_limit))
        return combined
    raise ValueError(f"Unknown suite '{name}'. Use riddles | gsm8k-lite | all.")
