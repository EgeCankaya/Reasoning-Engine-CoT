"""Typed configuration schema for training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml


def _get(mapping: Mapping[str, Any], key: str, default: Any) -> Any:
    return mapping[key] if key in mapping else default


@dataclass
class LoRAConfig:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LoRAConfig":
        return cls(
            r=int(_get(data, "r", cls.r)),
            lora_alpha=int(_get(data, "lora_alpha", cls.lora_alpha)),
            lora_dropout=float(_get(data, "lora_dropout", cls.lora_dropout)),
            target_modules=list(_get(data, "target_modules", cls().target_modules)),
        )


@dataclass
class TrainingConfig:
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    max_steps: int = 1000
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    optim: str = "adamw_8bit"
    lr_scheduler_type: str = "cosine"
    bf16: str | bool = "auto"
    fp16: str | bool = "auto"
    load_best_model_at_end: bool = True
    early_stopping_patience: int = 3

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        return cls(
            per_device_train_batch_size=int(_get(data, "per_device_train_batch_size", cls.per_device_train_batch_size)),
            gradient_accumulation_steps=int(_get(data, "gradient_accumulation_steps", cls.gradient_accumulation_steps)),
            warmup_ratio=float(_get(data, "warmup_ratio", cls.warmup_ratio)),
            max_steps=int(_get(data, "max_steps", cls.max_steps)),
            learning_rate=float(_get(data, "learning_rate", cls.learning_rate)),
            weight_decay=float(_get(data, "weight_decay", cls.weight_decay)),
            max_grad_norm=float(_get(data, "max_grad_norm", cls.max_grad_norm)),
            logging_steps=int(_get(data, "logging_steps", cls.logging_steps)),
            eval_steps=int(_get(data, "eval_steps", cls.eval_steps)),
            save_steps=int(_get(data, "save_steps", cls.save_steps)),
            optim=str(_get(data, "optim", cls.optim)),
            lr_scheduler_type=str(_get(data, "lr_scheduler_type", cls.lr_scheduler_type)),
            bf16=_get(data, "bf16", cls.bf16),
            fp16=_get(data, "fp16", cls.fp16),
            load_best_model_at_end=bool(_get(data, "load_best_model_at_end", cls.load_best_model_at_end)),
            early_stopping_patience=int(_get(data, "early_stopping_patience", cls.early_stopping_patience)),
        )


@dataclass
class StageConfig:
    max_steps: int = 0
    learning_rate: float = 0.0
    beta: float | None = None
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], defaults: "StageConfig") -> "StageConfig":
        return cls(
            max_steps=int(_get(data, "max_steps", defaults.max_steps)),
            learning_rate=float(_get(data, "learning_rate", defaults.learning_rate)),
            beta=_get(data, "beta", defaults.beta),
            enabled=bool(_get(data, "enabled", defaults.enabled)),
        )


@dataclass
class StagesConfig:
    enabled: bool = True
    format_learning: StageConfig = field(default_factory=lambda: StageConfig(max_steps=200, learning_rate=5e-5))
    reasoning_depth: StageConfig = field(default_factory=lambda: StageConfig(max_steps=600, learning_rate=1e-4))
    dpo_refinement: StageConfig = field(
        default_factory=lambda: StageConfig(max_steps=200, learning_rate=1e-4, beta=0.1, enabled=True)
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StagesConfig":
        defaults = cls()
        return cls(
            enabled=bool(_get(data, "enabled", defaults.enabled)),
            format_learning=StageConfig.from_dict(data.get("format_learning", {}), defaults=defaults.format_learning),
            reasoning_depth=StageConfig.from_dict(data.get("reasoning_depth", {}), defaults=defaults.reasoning_depth),
            dpo_refinement=StageConfig.from_dict(data.get("dpo_refinement", {}), defaults=defaults.dpo_refinement),
        )


@dataclass
class TrainingSettings:
    model_name: str = "models/base"
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    stages: StagesConfig | None = field(default_factory=StagesConfig)
    output_dir: str = "models/adapters"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TrainingSettings":
        return cls(
            model_name=str(_get(data, "model_name", cls.model_name)),
            max_seq_length=int(_get(data, "max_seq_length", cls.max_seq_length)),
            load_in_4bit=bool(_get(data, "load_in_4bit", cls.load_in_4bit)),
            lora=LoRAConfig.from_dict(_get(data, "lora", {})),
            training=TrainingConfig.from_dict(_get(data, "training", {})),
            stages=StagesConfig.from_dict(_get(data, "stages", {})) if _get(data, "stages", None) is not None else None,
            output_dir=str(_get(data, "output_dir", cls.output_dir)),
        )


def load_settings(path: str | Path) -> TrainingSettings:
    """Load training settings from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: MutableMapping[str, Any] = yaml.safe_load(handle) or {}
    if not isinstance(raw, MutableMapping):
        raise ValueError("Config file must contain a mapping at the root.")
    return TrainingSettings.from_mapping(raw)
