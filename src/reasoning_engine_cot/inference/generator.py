"""Text generation and response parsing for the reasoning engine."""

from __future__ import annotations

import re
from collections.abc import Generator
from contextlib import suppress
from typing import Any, Literal, TypedDict, cast

import torch

from .loader import ModelLoader

THINKING_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Generic tag-ish removal. Handles both well-formed tags (<tag ...>) and some malformed ones (<tag ...).
ANGLE_TAG_RE = re.compile(r"</?[A-Za-z0-9_]+(?:\s[^>]*)?>")
MALFORMED_ANGLE_TAG_RE = re.compile(r"</?[A-Za-z0-9_]+\b")

# Special token syntax like <|reflection_id|> or <|eot_id|>
SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")

# Marker words some models emit (sometimes with extra spaces / underscores).
_MARKER_WORDS = [
    "initial_analysis",
    "conscious_thought",
    "step_by_step",
    "reflection",
    "feeling",
    "self_improvement",
    "reflection_on_experience",
    "elaboration",
    "solution",
    "final_answer",
]


def _spaced_word_pattern(word: str) -> str:
    # Match a marker even if the model inserts spaces/underscores between characters.
    # Example: "initial_analysis" should match "ini ial_analysis" and "ini t ial a nalysis".
    letters = re.sub(r"[^A-Za-z0-9]", "", word)
    parts: list[str] = []
    for ch in letters:
        parts.append(re.escape(ch))
        parts.append(r"[\s_]*")
    # Remove the final [\s_]* that we appended after the last character.
    joined = "".join(parts)
    suffix = r"[\s_]*"
    if joined.endswith(suffix):
        joined = joined[: -len(suffix)]
    return joined


MARKER_WORD_RE = re.compile(r"(?is)\b(?:" + "|".join(_spaced_word_pattern(w) for w in _MARKER_WORDS) + r")\b")


class ReasoningGenerator:
    """Generate model outputs and parse thinking/answer segments."""

    def __init__(self, loader: ModelLoader, max_new_tokens: int = 512, temperature: float = 0.3, top_p: float = 0.9):
        self.loader = loader
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.model, self.tokenizer = loader.load()
        # Unsloth returns HF model/tokenizer objects; keep them as `Any` to avoid requiring stubs.
        self.model = cast(Any, self.model)
        self.tokenizer = cast(Any, self.tokenizer)

    @staticmethod
    def parse_response(text: str) -> dict[str, str | None]:
        thinking_match = THINKING_RE.search(text)
        answer_match = ANSWER_RE.search(text)
        return {
            "thinking": thinking_match.group(1).strip() if thinking_match else None,
            "answer": answer_match.group(1).strip() if answer_match else None,
        }

    def _format_prompt(self, question: str) -> str:
        instruction = (
            "You are a reasoning assistant.\n"
            "Return your reasoning inside <thinking>...</thinking> and your final answer inside <answer>...</answer>.\n"
            "Do not output any other tags, IDs, or metadata."
        )

        # Prefer tokenizer chat templates when available (Qwen / Llama / etc).
        # Hard-coding <|start_header_id|> tokens breaks non-Llama models and can cause junk outputs.
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": question.strip()},
            ]
            with suppress(Exception):
                return str(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

        return f"{instruction}\n\nQuestion: {question.strip()}\n\nAnswer:"

    @staticmethod
    def _extract_partial_sections(text: str) -> tuple[str | None, str | None]:
        """Best-effort extraction of thinking/answer, even if tags are not closed."""
        thinking: str | None = None
        answer: str | None = None

        # Heuristic: many datasets/models emit a <thinking> block and then plain answer text
        # without wrapping it in <answer>. If we see </thinking> but no <answer>, treat the
        # remainder as the answer. Also handle the case where </thinking> appears without
        # a clean opening <thinking>.
        if "<answer>" not in text and "</thinking>" in text:
            before_close, after_close = text.split("</thinking>", 1)
            answer = after_close
            thinking = before_close.split("<thinking>", 1)[1] if "<thinking>" in before_close else before_close

        if "<thinking>" in text:
            after = text.split("<thinking>", 1)[1]
            # Prefer proper closing tag, otherwise stop at answer tag if present.
            if "</thinking>" in after:
                thinking = after.split("</thinking>", 1)[0]
            elif "<answer>" in after:
                thinking = after.split("<answer>", 1)[0]
            else:
                thinking = after

        if "<answer>" in text:
            after = text.split("<answer>", 1)[1]
            answer = after.split("</answer>", 1)[0] if "</answer>" in after else after

        def _strip_noise(s: str) -> str:
            # Remove special tokens like <|reflection_id|>, <|start_header_id|>, <|eot_id|>
            s = SPECIAL_TOKEN_RE.sub("", s)
            # Remove generic XML-ish tags and some malformed variants
            s = ANGLE_TAG_RE.sub("", s)
            s = MALFORMED_ANGLE_TAG_RE.sub("", s)
            # Remove marker words even if spaced out (e.g., "ini ial_analysis")
            s = MARKER_WORD_RE.sub("", s)
            # Normalize whitespace
            s = re.sub(r"[ \t]+", " ", s)
            s = re.sub(r"\n{3,}", "\n\n", s)
            return s.strip()

        def _is_only_control_markers(s: str) -> bool:
            # Some Qwen-style outputs degrade into only marker words after we strip <|...|>.
            # If there's no real content, treat it as empty so we don't show junk in the UI.
            marker_only = re.compile(
                r"(?is)^\s*(?:"
                r"initial_analysis|conscious_thought|self_improvement|reflection|feeling|"
                r"elaboration|solution"
                r")(?:\s+|\s*$)+"
            )
            return bool(marker_only.match(s)) and len(re.sub(r"\s+", "", s)) < 500

        def _extract_final_answer(s: str) -> str:
            # Prefer explicit markers in verbose completions.
            marker_re = re.compile(r"(?is)(?:^|\n)\s*(final\s*answer|answer|solution)\s*:\s*(.+?)\s*$")
            matches = list(marker_re.finditer(s))
            if matches:
                s = matches[-1].group(2).strip()
            lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
            if not lines:
                return s.strip()

            # Drop common "metadata" lines some models emit (esp. CoT datasets / safety wrappers).
            meta_prefix = re.compile(
                r"(?is)^\s*(?:"
                r"task|approach|complexity|reflection|analysis|thought|"
                r"confidence|emotional\s*response|feeling|self-?improvement|"
                r"to\s+improve|notes?|explanation"
                r")\s*:"
            )
            filtered = [ln for ln in lines if not meta_prefix.match(ln)]

            # Prefer a short, non-metadata final line (usually the actual answer).
            candidates = filtered or lines
            # Drop list/bullet scaffolding.
            candidates = [ln.lstrip("-• ").strip() for ln in candidates if ln.strip()]

            # If we have multiple candidate lines, pick the last "answer-looking" one.
            answerish = [ln for ln in candidates if ":" not in ln and len(ln) <= 200]
            if answerish:
                return answerish[-1].strip()

            # Otherwise, fall back to the last non-empty line.
            return candidates[-1].strip()

        def _clean_thinking(x: str | None) -> str | None:
            if x is None:
                return None
            cleaned = _strip_noise(
                x.replace("<thinking>", "").replace("</thinking>", "").replace("<answer>", "").replace("</answer>", "")
            )
            if not cleaned or _is_only_control_markers(cleaned):
                return None
            return cleaned

        def _clean_answer(x: str | None) -> str | None:
            if x is None:
                return None
            cleaned = _strip_noise(
                x.replace("<thinking>", "").replace("</thinking>", "").replace("<answer>", "").replace("</answer>", "")
            )
            if not cleaned or _is_only_control_markers(cleaned):
                return None
            cleaned = _extract_final_answer(cleaned)
            if not cleaned or _is_only_control_markers(cleaned):
                return None
            return cleaned or None

        return _clean_thinking(thinking), _clean_answer(answer)

    def generate(self, question: str, stream: bool = False) -> str | Generator[str, None, None]:
        prompt = self._format_prompt(question)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
            )
        prompt_length = inputs["input_ids"].shape[1]
        # Decode only the generated completion (not the prompt) to avoid leaking chat-template
        # role labels like "system/user/assistant" into parsing.
        text = str(self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True))
        if not stream:
            return text
        return self._stream_tokens(outputs, prompt_length)

    def _stream_tokens(self, outputs: Any, prompt_length: int) -> Generator[str, None, None]:
        """Yield generated tokens after the prompt (naive streaming)."""
        for token_id in outputs[0][prompt_length:]:
            yield str(self.tokenizer.decode(token_id, skip_special_tokens=True))

    class StreamChunk(TypedDict):
        type: Literal["thinking", "answer"]
        content: str
        done: bool

    def stream_with_parsing(self, question: str) -> Generator[StreamChunk, None, None]:
        buffer = ""
        last_thinking: str | None = None
        last_answer: str | None = None

        for token in self.generate(question, stream=True):
            buffer += token

            thinking, answer = self._extract_partial_sections(buffer)

            if thinking is not None and thinking != last_thinking:
                last_thinking = thinking
                yield {"type": "thinking", "content": thinking, "done": False}

            if answer is not None and answer != last_answer:
                last_answer = answer
                # Only mark done once it looks complete (closed tag or end token), otherwise keep streaming updates.
                done = ("</answer>" in buffer) or ("<|eot_id|>" in buffer)
                yield {"type": "answer", "content": answer, "done": done}
                if done:
                    return

        # Stream ended. Prefer a parsed answer if we have one; only fall back to raw text
        # when the model never produced an <answer> section.
        if last_answer is not None:
            yield {"type": "answer", "content": last_answer, "done": True}
            return

        if buffer.strip():
            # Don't dump raw text (which can be mostly control tokens). Re-run extraction/cleanup.
            thinking, answer = self._extract_partial_sections(buffer)
            if answer is not None:
                yield {"type": "answer", "content": answer, "done": True}
                return
            cleaned = re.sub(SPECIAL_TOKEN_RE, "", buffer)
            cleaned = re.sub(ANGLE_TAG_RE, "", cleaned)
            cleaned = re.sub(MALFORMED_ANGLE_TAG_RE, "", cleaned).strip()
            cleaned = re.sub(MARKER_WORD_RE, "", cleaned).strip()
            if cleaned:
                yield {"type": "answer", "content": cleaned, "done": True}
            else:
                yield {"type": "answer", "content": "No answer detected.", "done": True}
