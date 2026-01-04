"""Fine-tuning pipeline.

- Uses Unsloth QLoRA when available (4-bit).
- Falls back to standard Transformers + PEFT LoRA on Windows when bitsandbytes/Unsloth is not usable.

This file avoids importing Unsloth at module import time so unit tests and CPU-only envs can import
`reasoning_engine_cot.training` without crashing.
"""

from __future__ import annotations

import argparse
import builtins
import logging
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict

from .config_schema import TrainingSettings, load_settings

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class QLoRATrainer:
    """Encapsulate model setup, LoRA application, and training."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.settings: TrainingSettings = load_settings(self.config_path)
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def setup_model(self) -> None:
        model_name = self.settings.model_name
        load_in_4bit = bool(self.settings.load_in_4bit)
        max_seq_length = int(self.settings.max_seq_length)

        LOGGER.info("Loading base model %s (4bit=%s)", model_name, load_in_4bit)

        if load_in_4bit:
            try:
                import unsloth  # noqa: F401
                from unsloth import FastLanguageModel

                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name,
                    max_seq_length=max_seq_length,
                    load_in_4bit=True,
                )

                self.model = FastLanguageModel.get_peft_model(
                    self.model,
                    r=int(self.settings.lora.r),
                    target_modules=self.settings.lora.target_modules,
                    lora_alpha=int(self.settings.lora.lora_alpha),
                    lora_dropout=float(self.settings.lora.lora_dropout),
                    bias="none",
                    use_gradient_checkpointing=True,
                )
                return
            except Exception as exc:
                if sys.platform == "win32":
                    LOGGER.warning(
                        "Unsloth/4-bit setup failed on Windows (%s). Falling back to non-4bit HF loading. "
                        "If your base model is a *bnb-4bit* repo, download a non-quantized base model for training.",
                        type(exc).__name__,
                    )
                else:
                    raise

        # Fallback: standard transformers + PEFT LoRA (no QLoRA).
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        lora_cfg = LoraConfig(
            r=self.settings.lora.r,
            lora_alpha=self.settings.lora.lora_alpha,
            lora_dropout=self.settings.lora.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.settings.lora.target_modules,
        )

        self.model = get_peft_model(model, lora_cfg)
        self.tokenizer = tokenizer

    def _build_training_args(self, output_dir: str, has_eval: bool) -> Any:
        from transformers import TrainingArguments

        cfg = self.settings.training
        bf16 = cfg.bf16
        fp16 = cfg.fp16
        eval_strategy = "steps" if has_eval else "no"

        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            warmup_ratio=cfg.warmup_ratio,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            logging_steps=cfg.logging_steps,
            optim=cfg.optim,
            lr_scheduler_type=cfg.lr_scheduler_type,
            evaluation_strategy=eval_strategy,
            eval_steps=cfg.eval_steps if has_eval else None,
            save_strategy="steps",
            save_steps=cfg.save_steps,
            bf16=torch.cuda.is_bf16_supported() if bf16 == "auto" else bool(bf16),
            fp16=not torch.cuda.is_bf16_supported() if fp16 == "auto" else bool(fp16),
            save_total_limit=2,
            report_to="none",
            load_best_model_at_end=cfg.load_best_model_at_end if has_eval else False,
            metric_for_best_model="eval_loss" if has_eval else None,
        )

    def _prepare_datasets(self, dataset: Dataset | DatasetDict) -> tuple[Dataset, Dataset | None]:
        if isinstance(dataset, DatasetDict):
            train_split = dataset["train"]
            eval_split = dataset.get("validation") or dataset.get("test")
            return train_split, eval_split

        split = dataset.train_test_split(test_size=0.1, seed=42)
        return split["train"], split["test"]

    def _build_callbacks(self, has_eval: bool) -> list[Any]:
        from transformers import EarlyStoppingCallback

        callbacks: list[Any] = []
        patience = self.settings.training.early_stopping_patience
        if has_eval and patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))
        return callbacks

    def train(self, dataset: Dataset | DatasetDict) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call setup_model() first.")

        # Unsloth's compiled trainer code may reference `psutil` without importing it.
        # Ensure it's available to avoid NameError on Windows.
        with suppress(Exception):  # pragma: no cover
            import psutil

            builtins.psutil = psutil

        from trl import SFTTrainer

        output_dir = self.settings.output_dir
        train_split, eval_split = self._prepare_datasets(dataset)
        has_eval = eval_split is not None
        training_args = self._build_training_args(output_dir=output_dir, has_eval=has_eval)

        LOGGER.info("Starting training with %d samples", len(train_split))

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_split,
            eval_dataset=eval_split,
            dataset_text_field="text",
            args=training_args,
            packing=False,
            callbacks=self._build_callbacks(has_eval),
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
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Saving merged model to %s", output_path)
        self.model.save_pretrained(output_path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune with QLoRA (Unsloth) or LoRA (fallback).")
    parser.add_argument(
        "--config", type=str, default="src/reasoning_engine_cot/training/config.yaml", help="Path to YAML config."
    )
    parser.add_argument("--dataset_name", type=str, default=None, help="HF dataset to download.")
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

    # Force stable behavior on Windows.
    if sys.platform == "win32":
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    trainer = QLoRATrainer(config_path=args.config)
    trainer.setup_model()

    if not args.dataset_name:
        raise ValueError("A dataset_name must be provided to run training.")

    from reasoning_engine_cot.data import CoTFormatter, DatasetDownloader

    downloader = DatasetDownloader()
    dataset = downloader.download(args.dataset_name)
    dataset = CoTFormatter().format_dataset(dataset)

    if args.add_instruction_prefix:
        prefix = (
            "You are a reasoning assistant. Always respond with <thinking>...</thinking> followed by "
            "<answer>...</answer>. Stay concise and keep the final answer inside <answer>."
        )

        def _add_prefix(example: dict[str, Any]) -> dict[str, Any]:
            return {"text": f"{prefix}\n\n{example['text']}"}

        dataset = dataset.map(_add_prefix)

    trainer.train(dataset)
    trainer.save_adapters(trainer.settings.output_dir)

    if args.merge_output:
        trainer.merge_and_save(args.merge_output)


if __name__ == "__main__":  # pragma: no cover
    main()
