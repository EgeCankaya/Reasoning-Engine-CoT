"""Run a tiny smoke check for tag adherence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

from reasoning_engine_cot.inference import ModelLoader, ReasoningGenerator


@dataclass
class SmokeResult:
    prompt: str
    thinking_ok: bool
    answer_ok: bool
    raw: str


def score_output(text: str) -> tuple[bool, bool]:
    return ("<thinking>" in text and "</thinking>" in text, "<answer>" in text and "</answer>" in text)


def run_smoke(prompts: List[str], use_adapters: bool) -> List[SmokeResult]:
    loader = ModelLoader(adapter_path="models/adapters" if use_adapters else None)
    generator = ReasoningGenerator(loader)
    results: List[SmokeResult] = []
    for p in prompts:
        out = generator.generate(p, stream=False)
        thinking_ok, answer_ok = score_output(out)
        results.append(SmokeResult(prompt=p, thinking_ok=thinking_ok, answer_ok=answer_ok, raw=out))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small tag-adherence smoke test.")
    parser.add_argument("--adapters", action=argparse.BooleanOptionalAction, default=False, help="Use adapters if present.")
    args = parser.parse_args()

    prompts = [
        "What has to be broken before you can use it?",
        "If I have 3 apples and eat one, how many remain?",
        "Solve: A farmer has 12 eggs and sells 5. How many are left?",
    ]

    results = run_smoke(prompts, use_adapters=args.adapters)
    total = len(results)
    thinking_hits = sum(r.thinking_ok for r in results)
    answer_hits = sum(r.answer_ok for r in results)

    print(f"Adapter mode: {args.adapters}")
    print(f"Thinking tags: {thinking_hits}/{total} | Answer tags: {answer_hits}/{total}")
    for r in results:
        print("-" * 60)
        print(f"Prompt: {r.prompt}")
        print(f"Thinking tag ok: {r.thinking_ok}, Answer tag ok: {r.answer_ok}")
        print(f"Output: {r.raw[:500]}")


if __name__ == "__main__":
    main()




