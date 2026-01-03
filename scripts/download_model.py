"""Download an open model into models/base without requiring a HF token."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a model into models/base.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        help="Hugging Face model repo to download (must be public/tokenless or token configured).",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="models/base",
        help="Destination directory for the downloaded model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # Enable faster downloads if hf-transfer is available.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=dst,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"✅ Downloaded {args.repo_id} to {dst}")


if __name__ == "__main__":
    main()






