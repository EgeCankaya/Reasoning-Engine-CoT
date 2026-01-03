"""Model loading utilities with optional LoRA adapters."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=2)
def _load_cached(model_name: str, adapter_path: str | None, max_seq_length: int, load_in_4bit: bool) -> tuple[Any, Any]:
    """Load the model and tokenizer, applying adapters if provided.

    Cached as a pure function to avoid `lru_cache` on instance methods (which can keep `self` alive).
    """
    # NOTE: Unsloth requires a GPU/accelerator. Importing it at module import time
    # breaks CPU-only environments (including unit tests that only need parsing).
    from peft import PeftModel
    from unsloth import FastLanguageModel

    base_path = model_name
    if not base_path:
        raise RuntimeError("MODEL_NAME is empty; set MODEL_NAME to your base model path or HF repo.")

    # Load base model first.
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            base_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        # Common root cause on Windows: torch can't see CUDA in the Streamlit process.
        try:
            import torch

            cuda_info = f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}"
        except Exception as torch_exc:  # pragma: no cover
            cuda_info = f"torch import failed: {torch_exc}"
        raise RuntimeError(
            f"Failed to load base model '{base_path}'. ({cuda_info})\n"
            "Set MODEL_NAME to a valid local path (e.g., models/base) or an HF repo you can access.\n"
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc

    # If no adapters, return base.
    if not adapter_path:
        return model, tokenizer

    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise RuntimeError(f"Adapter path '{adapter_dir}' does not exist. Train adapters first or switch to Base mode.")

    try:
        model = PeftModel.from_pretrained(model, adapter_dir)
    except Exception as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            f"Failed to load adapters from '{adapter_dir}'. Ensure adapters were trained against the same base MODEL_NAME."
        ) from exc

    return model, tokenizer


class ModelLoader:
    """Load base or fine-tuned models, caching instances to avoid reloads."""

    def __init__(
        self,
        model_name: str | None = None,
        adapter_path: str | None = None,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
    ) -> None:
        default_model = "unsloth/llama-3-8b-instruct-bnb-4bit"
        # Prefer explicit arg, then env, then default.
        self.model_name = model_name or os.getenv("MODEL_NAME") or default_model
        self.adapter_path = adapter_path
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit

    def load(self) -> tuple[Any, Any]:
        return _load_cached(self.model_name, self.adapter_path, self.max_seq_length, self.load_in_4bit)
