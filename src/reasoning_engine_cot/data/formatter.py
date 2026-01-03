"""Formatting utilities for Chain-of-Thought training data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from datasets import Dataset, DatasetDict

_THINKING_OPEN = "<thinking>"
_THINKING_CLOSE = "</thinking>"
_ANSWER_OPEN = "<answer>"
_ANSWER_CLOSE = "</answer>"


LLAMA3_TEMPLATE = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
<thinking>
{reasoning}
</thinking>
<answer>
{answer}
</answer><|eot_id|>
"""


QUESTION_KEYS = ("question", "instruction", "input", "prompt")
REASONING_KEYS = ("reasoning", "rationale", "cot", "steps", "chain_of_thought")
ANSWER_KEYS = ("final_answer", "answer", "output", "response")


@dataclass
class CoTFormatter:
    """Format raw samples into the Llama 3 chat template with thinking/answer tags."""

    template: str = LLAMA3_TEMPLATE

    def format_sample(self, question: str, reasoning: str, answer: str) -> str:
        """Format a single sample."""
        if not question or not reasoning or not answer:
            raise ValueError("question, reasoning, and answer must be non-empty strings.")

        return self.template.format(
            question=question.strip(),
            reasoning=reasoning.strip(),
            answer=answer.strip(),
        )

    def _infer_keys(self, sample: dict, question_key: Optional[str], reasoning_key: Optional[str], answer_key: Optional[str]) -> tuple[str, str, str]:
        qk = question_key or next((k for k in QUESTION_KEYS if k in sample), None)
        rk = reasoning_key or next((k for k in REASONING_KEYS if k in sample), None)
        ak = answer_key or next((k for k in ANSWER_KEYS if k in sample), None)

        if not qk or not rk or not ak:
            raise KeyError(
                "Could not infer required fields. "
                f"Found keys: {list(sample.keys())}. "
                f"Expected question in {QUESTION_KEYS}, reasoning in {REASONING_KEYS}, answer in {ANSWER_KEYS}."
            )
        return qk, rk, ak

    def format_dataset(
        self,
        dataset: Dataset | DatasetDict,
        question_key: Optional[str] = None,
        reasoning_key: Optional[str] = None,
        answer_key: Optional[str] = None,
    ) -> Dataset | DatasetDict:
        """Return a dataset with a new 'text' column containing formatted prompts."""

        def _format(example: dict) -> dict:
            # Special-case: datasets like isaiahbjork/chain-of-thought ship as {prompt, response}
            # where response contains a <thinking>...</thinking> block and then the final answer text.
            if "prompt" in example and "response" in example and (reasoning_key is None):
                question = str(example["prompt"])
                response = str(example["response"])

                thinking = ""
                answer = ""

                if _THINKING_OPEN in response and _THINKING_CLOSE in response:
                    after_open = response.split(_THINKING_OPEN, 1)[1]
                    thinking = after_open.split(_THINKING_CLOSE, 1)[0]
                    answer = after_open.split(_THINKING_CLOSE, 1)[1]
                else:
                    # If the dataset doesn't provide a thinking block, treat the full response as both.
                    thinking = response
                    answer = response

                # If an explicit <answer> tag is present, prefer that.
                if _ANSWER_OPEN in response:
                    after = response.split(_ANSWER_OPEN, 1)[1]
                    answer = after.split(_ANSWER_CLOSE, 1)[0] if _ANSWER_CLOSE in after else after

                # Normalize + ensure non-empty fields (some rows have empty tail after </thinking>).
                question = question.strip()
                thinking = str(thinking).strip()
                answer = str(answer).strip()
                if not thinking:
                    thinking = response.strip() or " "
                if not answer:
                    # Prefer the tail after </thinking>, otherwise fall back to response/thinking.
                    tail = ""
                    if _THINKING_CLOSE in response:
                        tail = response.split(_THINKING_CLOSE, 1)[1].strip()
                    answer = tail or response.strip() or thinking

                return {"text": self.format_sample(question, thinking, answer)}

            qk, rk, ak = self._infer_keys(example, question_key, reasoning_key, answer_key)
            return {"text": self.format_sample(example[qk], example[rk], example[ak])}

        if isinstance(dataset, DatasetDict):
            return DatasetDict({split: ds.map(_format) for split, ds in dataset.items()})
        return dataset.map(_format)

    def validate_samples(self, samples: Iterable[dict], question_key: str, reasoning_key: str, answer_key: str) -> None:
        """Validate that required keys exist and contain text."""
        for idx, sample in enumerate(samples):
            for key, label in ((question_key, "question"), (reasoning_key, "reasoning"), (answer_key, "answer")):
                if key not in sample or not str(sample[key]).strip():
                    raise ValueError(f"Sample {idx} missing or empty '{label}' field (key '{key}')")




















