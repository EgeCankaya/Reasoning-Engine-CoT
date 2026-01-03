"""Dataset download helpers."""

from __future__ import annotations

from dataclasses import dataclass

from datasets import DatasetDict, load_dataset

SUPPORTED_DATASETS = {
    "isaiahbjork/chain-of-thought",
    "AlekseyKorshuk/chain-of-thoughts-chatml",
}


@dataclass
class DatasetDownloader:
    """Download Chain-of-Thought datasets from Hugging Face with caching."""

    cache_dir: str | None = None

    def download(self, dataset_name: str) -> DatasetDict:
        """Download a supported dataset and return a DatasetDict with splits.

        Args:
            dataset_name: Hugging Face dataset identifier.

        Raises:
            ValueError: If the dataset is not supported.
        """
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {sorted(SUPPORTED_DATASETS)}")

        return load_dataset(path=dataset_name, cache_dir=self.cache_dir)
