"""Run the Streamlit UI with sane defaults (Windows-friendly).

This is used by `make run` so contributors don't need `uv` installed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_model_name_env(repo_root: Path) -> None:
    # Prefer user-provided MODEL_NAME. Otherwise, if the user already downloaded a model
    # to models/base, use it automatically.
    if os.getenv("MODEL_NAME"):
        return
    candidate = repo_root / "models" / "base"
    if candidate.exists():
        os.environ["MODEL_NAME"] = str(candidate)


def _ensure_compiler_available() -> None:
    # Triton on Windows needs a C/C++ compiler (MSVC cl.exe) available on PATH,
    # typically via "x64 Native Tools Command Prompt for VS".
    if sys.platform != "win32":
        return
    if os.getenv("CC"):
        return
    if shutil.which("cl.exe"):
        os.environ["CC"] = "cl.exe"
        return

    raise RuntimeError(
        "No C compiler found (cl.exe). Install VS Build Tools (Desktop development with C++) "
        "and run this command from the 'x64 Native Tools Command Prompt for VS 2022' "
        "(or set CC to your cl.exe path)."
    )


def main() -> None:
    repo_root = _repo_root()
    os.chdir(repo_root)

    _ensure_model_name_env(repo_root)
    _ensure_compiler_available()

    model_name = os.getenv("MODEL_NAME", "")
    cc = os.getenv("CC", "not set")
    try:
        import torch

        torch_info = f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}"
    except Exception as exc:  # pragma: no cover
        torch_info = f"torch import failed: {exc}"

    print(f"[run_ui] MODEL_NAME={model_name or '(not set)'}")
    print(f"[run_ui] CC={cc}")
    print(f"[run_ui] {torch_info}")

    cmd = [sys.executable, "-m", "streamlit", "run", "src/reasoning_engine_cot/ui/app.py"]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()




