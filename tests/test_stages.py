from dataclasses import dataclass

from datasets import Dataset

from reasoning_engine_cot.data.curriculum import CurriculumSorter
from reasoning_engine_cot.data.filters import QualityFilter
from reasoning_engine_cot.training.config_schema import StageConfig
from reasoning_engine_cot.training.stages import TrainingStageManager


@dataclass
class DummyTraining:
    max_steps: int
    learning_rate: float


@dataclass
class DummySettings:
    training: DummyTraining


class DummyTrainer:
    def __init__(self) -> None:
        self.settings = DummySettings(training=DummyTraining(max_steps=1000, learning_rate=0.001))
        self.last_max_steps: int | None = None

    def train(self, dataset) -> None:  # noqa: ANN001
        self.last_max_steps = self.settings.training.max_steps


def test_stage_manager_overrides_and_restores_training_params() -> None:
    trainer = DummyTrainer()
    manager = TrainingStageManager(
        trainer=trainer,
        quality_filter=QualityFilter(min_text_length=0, require_tags=False),
        curriculum=CurriculumSorter(),
    )

    dataset = Dataset.from_dict({"text": ["foo", "bar"]})
    stage_cfg = StageConfig(max_steps=5, learning_rate=0.01, enabled=True)

    result = manager.run_stage_1_format_learning(dataset, stage_cfg)

    assert result.skipped is False
    assert trainer.last_max_steps == 5
    assert trainer.settings.training.max_steps == 1000
