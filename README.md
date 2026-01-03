# Reasoning-Engine-CoT

Chain-of-Thought fine-tuning specialist for logic and math puzzles. Trains a quantized open model (tokenless by default) with QLoRA to enforce the structure:

```
<thinking> ... step-by-step reasoning ... </thinking>
<answer> final answer </answer>
```

## Features
- QLoRA fine-tuning pipeline with Unsloth (4-bit) targeting projection modules.
- Data downloader/formatter for open CoT datasets (Hugging Face).
- Streamlit UI with collapsible “Thinking Process” and highlighted answer.
- Inference utilities with parsing of `<thinking>` / `<answer>` tags (with robust fallbacks for malformed tags).
- Windows-friendly dependency set (with `bitsandbytes-windows` fallback).

## Quickstart (Windows, CUDA GPU, tokenless)
1) Install Python 3.9–3.12 and a CUDA-enabled PyTorch wheel for your GPU.
2) Install deps:
```bash
pip install -r requirements.txt
```
3) Download a tokenless open model into `models/base` (default: `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`):
```bash
python scripts/download_model.py
```
4) (Optional) Run unit tests:
```bash
pytest
```

## Training (QLoRA, Unsloth)
```bash
# Fine-tune on a supported dataset (adapters saved to models/adapters)
python -m reasoning_engine_cot.training.fine_tune --dataset_name isaiahbjork/chain-of-thought
```
Config: `src/reasoning_engine_cot/training/config.yaml` (LoRA rank/alpha, steps, LR, output dir). Training expects a dataset with a `text` field; the CLI will download and format supported datasets automatically. The default base model is `models/base` (set `MODEL_NAME` to override). Adapters are saved to `models/adapters`. Use `--merge_output models/merged` to export a merged model if desired.

## Inference
Use the base or adapter path (tokenless default: `models/base`):
```python
from reasoning_engine_cot.inference import ModelLoader, ReasoningGenerator

loader = ModelLoader(adapter_path="models/adapters")  # or None for base
generator = ReasoningGenerator(loader)
result = generator.generate("If I have 3 apples and eat one, how many remain?")
parsed = generator.parse_response(result)
```

## Streamlit UI
```bash
# From a VS Developer prompt (so cl.exe is on PATH)
make run
```
Toggle between Base vs. CoT Fine-Tuned, view streamed thinking in an expander, and the final answer in a highlighted block. The sidebar shows the active base model (defaults to `models/base` if present).

## Repository Structure
- `src/reasoning_engine_cot/data/` – dataset download & formatting
- `src/reasoning_engine_cot/training/` – QLoRA config and trainer
- `src/reasoning_engine_cot/inference/` – model loader and generator with parsing
- `src/reasoning_engine_cot/ui/` – Streamlit app
- `models/adapters/` – saved LoRA adapters (gitignored)
- `notebooks/` – exploration, training, evaluation stubs
- `tests/` – formatter and parser unit tests

## Evaluation Guidance
- Format adherence: regex check for `<thinking>` / `<answer>` tags (target ≥95%).
- Logic accuracy: compare against GSM8K/logic puzzles vs. base model.
- VRAM budget: keep training <14 GB on RTX 4070 Ti Super.
- Latency: aim <50 ms/token in inference.

## License
MIT
