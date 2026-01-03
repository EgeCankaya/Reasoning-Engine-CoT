"""Lite evaluation: format adherence, simple correctness, and latency."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from reasoning_engine_cot.inference import ModelLoader, ReasoningGenerator
from eval.suites import Example, load_suite
from reasoning_engine_cot.eval.metrics import (
    strict_adherence,
    recovered_adherence,
    tokens_stats,
    reset_peak_memory,
    peak_memory_mb,
)


# Backwards-compat alias for earlier tests/docs.
def compute_adherence(text: str) -> tuple[bool, bool]:
    return strict_adherence(text)


@dataclass
class EvalResult:
    example_id: str
    model: str
    use_adapters: bool
    thinking_ok: bool
    answer_ok: bool
    thinking_recovered: bool
    answer_recovered: bool
    answer_match: bool
    latency_ms: float
    tokens: int
    ms_per_token: float
    vram_mb: Optional[float]
    output: str


def answer_contains(output: str, expected: str) -> bool:
    return expected.lower() in output.lower()


def run_one(generator: ReasoningGenerator, ex: Example, use_adapters: bool, model_name: str) -> EvalResult:
    reset_peak_memory()
    t0 = time.perf_counter()
    out = generator.generate(ex.question, stream=False)
    latency_ms = (time.perf_counter() - t0) * 1000
    tokens, ms_per_token, _ = tokens_stats(generator.tokenizer, out, latency_ms)
    thinking_ok, answer_ok = strict_adherence(out)
    thinking_rec, answer_rec = recovered_adherence(out)
    match = answer_contains(out, ex.answer)
    vram_mb = peak_memory_mb()
    return EvalResult(
        example_id=ex.id,
        model=model_name,
        use_adapters=use_adapters,
        thinking_ok=thinking_ok,
        answer_ok=answer_ok,
        thinking_recovered=thinking_rec,
        answer_recovered=answer_rec,
        answer_match=match,
        latency_ms=latency_ms,
        tokens=tokens,
        ms_per_token=ms_per_token,
        vram_mb=vram_mb,
        output=out,
    )


def run_eval(use_adapters: bool, suite: str, gsm8k_limit: int) -> List[EvalResult]:
    examples = load_suite(suite, gsm8k_limit=gsm8k_limit)
    loader = ModelLoader(adapter_path="models/adapters" if use_adapters else None)
    generator = ReasoningGenerator(loader)
    model_name = f"{loader.model_name}{'+adapters' if use_adapters else ''}"
    return [run_one(generator, ex, use_adapters, model_name) for ex in examples]


def write_report(results: List[EvalResult], out_dir: Path, suite: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "report.csv"
    md_path = out_dir / "summary.md"

    def agg(filter_fn):
        subset = [r for r in results if filter_fn(r)]
        total = len(subset)
        if total == 0:
            return {"total": 0}
        return {
            "total": total,
            "thinking": sum(r.thinking_ok for r in subset),
            "answer_tag": sum(r.answer_ok for r in subset),
            "thinking_rec": sum(r.thinking_recovered for r in subset),
            "answer_rec": sum(r.answer_recovered for r in subset),
            "match": sum(r.answer_match for r in subset),
            "avg_latency": sum(r.latency_ms for r in subset) / total,
            "avg_tokens": sum(r.tokens for r in subset) / total,
            "avg_ms_per_token": sum(r.ms_per_token for r in subset) / total,
            "avg_vram": (
                sum(r.vram_mb for r in subset if r.vram_mb is not None) / max(
                    1, sum(1 for r in subset if r.vram_mb is not None)
                )
                if any(r.vram_mb is not None for r in subset)
                else None
            ),
        }

    agg_all = agg(lambda r: True)
    agg_base = agg(lambda r: not r.use_adapters)
    agg_ft = agg(lambda r: r.use_adapters)

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "example_id,model,use_adapters,thinking_ok,answer_ok,thinking_recovered,answer_recovered,"
            "answer_match,latency_ms,tokens,ms_per_token,vram_mb,output\n"
        )
        for r in results:
            escaped_output = r.output.replace('"', "'").replace("\n", " ")[:500]
            f.write(
                f"{r.example_id},{r.model},{r.use_adapters},{int(r.thinking_ok)},{int(r.answer_ok)},"
                f"{int(r.thinking_recovered)},{int(r.answer_recovered)},{int(r.answer_match)},"
                f"{r.latency_ms:.2f},{r.tokens},{r.ms_per_token:.4f},"
                f"{'' if r.vram_mb is None else f'{r.vram_mb:.2f}'},\"{escaped_output}\"\n"
            )

    def fmt_block(name: str, agg_res: dict) -> str:
        if agg_res.get("total", 0) == 0:
            return f"- {name}: no samples\n"
        return (
            f"- {name} (n={agg_res['total']}): "
            f"thinking {agg_res['thinking']}/{agg_res['total']} (recovered {agg_res['thinking_rec']}/{agg_res['total']}), "
            f"answer tags {agg_res['answer_tag']}/{agg_res['total']} (recovered {agg_res['answer_rec']}/{agg_res['total']}), "
            f"match {agg_res['match']}/{agg_res['total']}, "
            f"latency {agg_res['avg_latency']:.2f} ms, "
            f"ms/token {agg_res['avg_ms_per_token']:.4f}, "
            f"vram {('n/a' if agg_res['avg_vram'] is None else f'{agg_res['avg_vram']:.2f} MB')}"
        )

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Evaluation Summary ({suite})\n\n")
        f.write(fmt_block("All", agg_all) + "\n")
        f.write(fmt_block("Base", agg_base) + "\n")
        f.write(fmt_block("Fine-tuned", agg_ft) + "\n")

    # Examples (only if both base and fine-tuned exist)
    if agg_base.get("total", 0) and agg_ft.get("total", 0):
        examples_path = out_dir / "examples.md"
        # pair by example_id
        by_id = {}
        for r in results:
            by_id.setdefault(r.example_id, {})["ft" if r.use_adapters else "base"] = r
        pairs = [v for v in by_id.values() if "base" in v and "ft" in v]
        pairs = pairs[:5]
        with examples_path.open("w", encoding="utf-8") as f:
            f.write("# Examples (Base vs Fine-tuned)\n\n")
            for idx, p in enumerate(pairs, 1):
                base = p["base"]
                ft = p["ft"]
                f.write(f"## Example {idx}: {base.example_id}\n\n")
                f.write("**Base output:**\n\n")
                f.write(f"> {base.output[:1500]}\n\n")
                f.write("**Fine-tuned output:**\n\n")
                f.write(f"> {ft.output[:1500]}\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lite evaluation for base vs adapters.")
    parser.add_argument("--suite", type=str, default="riddles", help="riddles | gsm8k-lite | all")
    parser.add_argument("--gsm8k_limit", type=int, default=100, help="Limit for GSM8K-lite.")
    parser.add_argument("--adapters", action=argparse.BooleanOptionalAction, default=False, help="Use adapters.")
    parser.add_argument("--ab", action=argparse.BooleanOptionalAction, default=False, help="Run both base and adapters.")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to write reports (default reports/<timestamp>).")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("reports") / timestamp

    modes = [args.adapters]
    if args.ab:
        modes = [False, True]

    all_results: List[EvalResult] = []
    for mode in modes:
        all_results.extend(run_eval(use_adapters=mode, suite=args.suite, gsm8k_limit=args.gsm8k_limit))

    write_report(all_results, out_dir, suite=args.suite)
    print(f"✅ Wrote reports to {out_dir}")


if __name__ == "__main__":
    main()


