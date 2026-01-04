"""Multi-stage training orchestration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from datasets import Dataset, DatasetDict

from reasoning_engine_cot.data.curriculum import CurriculumSorter
from reasoning_engine_cot.data.filters import QualityFilter

from .config_schema import StageConfig
from .fine_tune import QLoRATrainer


@dataclass
class StageRunResult:
    stage: str
    skipped: bool


class TrainingStageManager:
    def __init__(
        self,
        trainer: QLoRATrainer,
        quality_filter: QualityFilter | None = None,
        curriculum: CurriculumSorter | None = None,
    ) -> None:
        self.trainer = trainer
        self.quality_filter = quality_filter or QualityFilter()
        self.curriculum = curriculum or CurriculumSorter()

    def _prepare(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        if isinstance(dataset, DatasetDict):
            filtered = DatasetDict({split: self.quality_filter.filter(ds) for split, ds in dataset.items()})
            return DatasetDict({split: self.curriculum.sort(ds) for split, ds in filtered.items()})
        return self.curriculum.sort(self.quality_filter.filter(dataset))

    def run_stage_1_format_learning(self, dataset: Dataset | DatasetDict, stage_cfg: StageConfig) -> StageRunResult:
        return self._run_stage("format_learning", dataset, stage_cfg)

    def run_stage_2_reasoning_depth(self, dataset: Dataset | DatasetDict, stage_cfg: StageConfig) -> StageRunResult:
        return self._run_stage("reasoning_depth", dataset, stage_cfg)

    def run_stage_3_dpo_refinement(self, dataset: Dataset | DatasetDict, stage_cfg: StageConfig) -> StageRunResult:
        return self._run_stage("dpo_refinement", dataset, stage_cfg)

    def _run_stage(self, name: str, dataset: Dataset | DatasetDict, stage_cfg: StageConfig) -> StageRunResult:
        if not stage_cfg.enabled:
            return StageRunResult(stage=name, skipped=True)

        prepared = self._prepare(dataset)
        original = deepcopy(self.trainer.settings.training)
        try:
            if stage_cfg.max_steps > 0:
                self.trainer.settings.training.max_steps = stage_cfg.max_steps
            if stage_cfg.learning_rate > 0:
                self.trainer.settings.training.learning_rate = stage_cfg.learning_rate
            self.trainer.train(prepared)
        finally:
            self.trainer.settings.training = original

        return StageRunResult(stage=name, skipped=False)
