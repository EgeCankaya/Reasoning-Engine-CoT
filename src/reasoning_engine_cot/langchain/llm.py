from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.language_models.llms import LLM

from reasoning_engine_cot.inference import ModelLoader, ReasoningGenerator


class ReasoningLLM(LLM):
    """LangChain LLM wrapper around ReasoningGenerator."""

    def __init__(
        self,
        use_adapters: bool = False,
        model_name: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> None:
        super().__init__()
        loader = ModelLoader(adapter_path="models/adapters" if use_adapters else None, model_name=model_name or None)
        self.generator = ReasoningGenerator(
            loader,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    @property
    def _llm_type(self) -> str:
        return "reasoning_cot"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        # stop is ignored; model handles its own stopping
        return self.generator.generate(prompt, stream=False)




