"""QLoRA fine-tuning pipeline using Unsloth and TRL."""

from __future__ import annotations

import os

# Windows + Triton version combos can break torch.compile/inductor. For training stability,
# disable Dynamo/Inductor compilation (Unsloth still provides speedups via its own kernels).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import unsloth  # noqa: F401  # must be imported before transformers/trl/peft for full optimizations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import builtins
import torch
import yaml
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments
from trl import SFTTrainer

from unsloth import FastLanguageModel

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QLoRATrainer:
    """Encapsulate model setup, LoRA application, and training."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.config = self.load_config(self.config_path)
        self.model = None
        self.tokenizer = None

    @staticmethod
    def load_config(path: str | Path) -> Dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def setup_model(self) -> None:
        model_name = self.config["model_name"]
        load_in_4bit = bool(self.config.get("load_in_4bit", True))
        max_seq_length = int(self.config.get("max_seq_length", 2048))

        LOGGER.info("Loading base model %s (4bit=%s)", model_name, load_in_4bit)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=max_seq_length,
             load_in_4bit=load_in_4bit,
        )

        lora_cfg = self.config.get("lora", {})
        LOGGER.info("Applying LoRA with config: %s", lora_cfg)
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=int(lora_cfg.get("r", 16)),
            target_modules=lora_cfg.get("target_modules"),
            lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
            lora_dropout=float(lora_cfg.get("lora_dropout", 0.0)),
            bias="none",
            use_gradient_checkpointing=True,
        )

    def _build_training_args(self, output_dir: str) -> TrainingArguments:
        cfg = self.config["training"]
        bf16 = cfg.get("bf16", "auto")
        fp16 = cfg.get("fp16", "auto")
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            warmup_steps=cfg["warmup_steps"],
            max_steps=cfg["max_steps"],
            learning_rate=cfg["learning_rate"],
            logging_steps=cfg["logging_steps"],
            optim=cfg["optim"],
            bf16=torch.cuda.is_bf16_supported() if bf16 == "auto" else bool(bf16),
            fp16=not torch.cuda.is_bf16_supported() if fp16 == "auto" else bool(fp16),
            save_total_limit=2,
            save_steps=cfg["logging_steps"] * 10,
            report_to="none",
        )

    def train(self, dataset: Dataset | DatasetDict) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")

        # Unsloth's compiled trainer code may reference `psutil` without importing it.
        # Ensure it's available to avoid NameError on Windows.
        try:  # pragma: no cover
            import psutil  # type: ignore

            builtins.psutil = psutil  # type: ignore[attr-defined]
        except Exception:
            pass

        output_dir = self.config.get("output_dir", "models/adapters")
        training_args = self._build_training_args(output_dir=output_dir)

        train_split = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
        LOGGER.info("Starting training with %d samples", len(train_split))

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_split,
            dataset_text_field="text",
            args=training_args,  
            packing=False,
        )
        trainer.train()

    def save_adapters(self, output_path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Saving LoRA adapters to %s", output_path)
        self.model.save_pretrained(output_path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_path)

    def merge_and_save(self, output_path: str | Path) -> None:
        """Optionally merge adapters into the base model for single-file export."""
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Merging adapters into base model and saving to %s", output_path)
        # For Unsloth models, save_pretrained merges if adapters are applied.
        self.model.save_pretrained(output_path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3 with QLoRA using Unsloth.")
    parser.add_argument("--config", type=str, default="src/reasoning_engine_cot/training/config.yaml", help="Path to YAML config.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Optional HF dataset to download.")
    parser.add_argument("--merge_output", type=str, default=None, help="Optional path to save merged base+adapters.")
    parser.add_argument(
        "--add_instruction_prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prepend a training-time instruction reminding the model to emit <thinking>/<answer> tags.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer = QLoRATrainer(config_path=args.config)
    trainer.setup_model()

    if args.dataset_name:
        from reasoning_engine_cot.data import CoTFormatter, DatasetDownloader

        downloader = DatasetDownloader()
        dataset = downloader.download(args.dataset_name)
        dataset = CoTFormatter().format_dataset(dataset)
    else:
        raise ValueError("A dataset_name must be provided to run training.")

    if args.add_instruction_prefix:
        prefix = (
            "You are a reasoning assistant. Always respond with <thinking>...</thinking> followed by "
            "<answer>...</answer>. Stay concise and keep the final answer inside <answer>."
        )

        def _add_prefix(example: dict) -> dict:
            return {"text": f"{prefix}\n\n{example['text']}"}

        dataset = dataset.map(_add_prefix)

    trainer.train(dataset)
    trainer.save_adapters(trainer.config.get("output_dir", "models/adapters"))

    if args.merge_output:
        trainer.merge_and_save(args.merge_output)


if __name__ == "__main__":  # pragma: no cover
    main()






