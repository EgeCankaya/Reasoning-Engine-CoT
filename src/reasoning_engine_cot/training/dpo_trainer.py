"""DPO trainer for reasoning quality refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import TrainingArguments
from trl import DPOTrainer


@dataclass
class DPOConfig:
    beta: float = 0.1
    max_steps: int = 200
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    logging_steps: int = 10
    save_steps: int = 100
    output_dir: str = "models/dpo"


class CoTDPOTrainer:
    def __init__(self, model: Any, tokenizer: Any, config: DPOConfig | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DPOConfig()

    def create_preference_pairs(self, dataset: Dataset) -> Dataset:
        required = {"prompt", "chosen", "rejected"}
        missing = required.difference(set(dataset.column_names))
        if missing:
            raise ValueError(f"Preference dataset missing required fields: {sorted(missing)}")
        return dataset

    def train(self, preference_dataset: Dataset) -> None:
        cfg = self.config
        args = TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            logging_steps=cfg.logging_steps,
            save_strategy="steps",
            save_steps=cfg.save_steps,
            report_to="none",
        )
        trainer = DPOTrainer(
            model=self.model,
            ref_model=None,
            args=args,
            beta=cfg.beta,
            train_dataset=preference_dataset,
            tokenizer=self.tokenizer,
        )
        trainer.train()

    def save(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
