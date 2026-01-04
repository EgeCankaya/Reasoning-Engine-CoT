from pathlib import Path

from reasoning_engine_cot.training.config_schema import LoRAConfig, TrainingSettings, load_settings


def test_load_settings_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text("", encoding="utf-8")

    settings = load_settings(config_path)
    assert isinstance(settings, TrainingSettings)
    assert settings.model_name
    assert settings.lora.r == 32
    assert settings.training.max_steps == 1000


def test_load_settings_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        """
model_name: \"custom/model\"
lora:
  r: 8
training:
  max_steps: 10
""",
        encoding="utf-8",
    )

    settings = load_settings(config_path)
    assert settings.model_name == "custom/model"
    assert settings.lora == LoRAConfig(r=8, lora_alpha=64, lora_dropout=0.05, target_modules=LoRAConfig().target_modules)
    assert settings.training.max_steps == 10
