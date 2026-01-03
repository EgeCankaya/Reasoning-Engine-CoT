"""Streamlit UI for the Logic Puzzle Solver."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from reasoning_engine_cot.inference import ModelLoader, ReasoningGenerator
from reasoning_engine_cot.langchain.llm import ReasoningLLM


@st.cache_resource(show_spinner=True)
def get_generator(use_adapters: bool, model_name: str) -> ReasoningGenerator:
    adapter_path = "models/adapters" if use_adapters else None
    loader = ModelLoader(model_name=model_name, adapter_path=adapter_path)
    return ReasoningGenerator(loader=loader)


def _result_key(model_label: str, prompt_text: str) -> str:
    # Normalize prompt so the same prompt maps to the same result key.
    return f"{model_label}::{prompt_text.strip()}"


def render() -> None:
    st.set_page_config(page_title="Reasoning Engine", page_icon="🧠", layout="wide")
    st.title("Reasoning Engine - Chain-of-Thought Specialist")
    st.write("Enter a riddle, math problem, or logic query. The model will reason before answering.")

    base_model = os.getenv("MODEL_NAME", "models/base")
    st.sidebar.write(f"Base model: `{base_model}`")
    if st.sidebar.button("Clear cached model"):
        # Streamlit can cache model instances across reruns. If MODEL_NAME or adapters changed,
        # clearing cache forces a clean reload.
        st.cache_resource.clear()
        st.rerun()
    model_choice = st.sidebar.radio("Model", ["CoT Fine-Tuned", "Base Model"])
    use_langchain = st.sidebar.checkbox("Use LangChain wrapper", value=False)
    use_history = st.sidebar.checkbox("Retain chat history", value=False)
    compare_mode = st.sidebar.checkbox("Compare Base vs CoT", value=True)
    use_adapters = model_choice == "CoT Fine-Tuned"
    prompt = st.text_area(
        "Your question",
        height=120,
        placeholder="e.g., What is so fragile that saying its name breaks it?",
        key="prompt_text",
    )

    if use_history and "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "results" not in st.session_state:
        # Map: "<Model Label>::<Prompt>" -> {"thinking": str|None, "answer": str|None, "raw": str|None}
        st.session_state.results = {}

    if use_adapters and not Path("models/adapters").exists():
        st.warning(
            "No adapters found in `models/adapters`. "
            "Train adapters (e.g., `python -m reasoning_engine_cot.training.fine_tune "
            "--dataset_name isaiahbjork/chain-of-thought`) or switch to Base."
        )

    def _run_once(model_label: str, adapters: bool, prompt_text: str) -> None:
        if adapters and not Path("models/adapters").exists():
            st.error(
                "No adapters found in `models/adapters`. Train or place LoRA weights there, or switch to Base Model."
            )
            return

        if not prompt_text.strip():
            st.warning("Please provide a question.")
            return

        # Build prompt with optional history
        full_prompt = prompt_text
        if use_history and st.session_state.get("chat_history"):
            history_lines = []
            for h in st.session_state.chat_history[-5:]:
                history_lines.append(f"Q: {h['q']}\nA: {h['a']}")
            history_block = "\n\n".join(history_lines)
            full_prompt = f"Previous exchanges:\n{history_block}\n\nCurrent question: {prompt_text}"

        if use_langchain:
            llm = ReasoningLLM(use_adapters=adapters)
            output = llm.invoke(full_prompt)
            thinking, answer = ReasoningGenerator._extract_partial_sections(output)
            st.session_state.results[_result_key(model_label, prompt_text)] = {
                "thinking": thinking,
                "answer": answer or output,
                "raw": output,
            }
        else:
            generator = get_generator(use_adapters=adapters, model_name=base_model)
            final_answer = None
            final_thinking = None
            raw_buffer = ""
            for chunk in generator.stream_with_parsing(full_prompt):
                if chunk["type"] == "thinking":
                    final_thinking = chunk["content"]
                if chunk["type"] == "answer":
                    final_answer = chunk["content"]
            # Best effort raw: we don't have access to internal buffer here; store the parsed answer/thinking.
            st.session_state.results[_result_key(model_label, prompt_text)] = {
                "thinking": final_thinking,
                "answer": final_answer,
                "raw": raw_buffer or None,
            }

        if use_history:
            # Store the answer for conversational context.
            res = st.session_state.results.get(_result_key(model_label, prompt_text)) or {}
            st.session_state.chat_history.append({"q": prompt_text, "a": res.get("answer") or ""})

    col_run1, col_run2 = st.columns(2)
    with col_run1:
        solve_clicked = st.button(f"Solve with {model_choice}")
    with col_run2:
        run_both_clicked = st.button("Run both (Base + CoT)")

    if solve_clicked:
        with st.spinner(f"Generating ({model_choice})..."):
            _run_once(model_choice, use_adapters, prompt)

    if run_both_clicked:
        if not prompt.strip():
            st.warning("Please provide a question.")
        else:
            with st.spinner("Generating (Base Model)..."):
                _run_once("Base Model", False, prompt)
            with st.spinner("Generating (CoT Fine-Tuned)..."):
                _run_once("CoT Fine-Tuned", True, prompt)

    def _render_result_box(model_label: str) -> None:
        res = st.session_state.results.get(_result_key(model_label, prompt)) if prompt.strip() else None
        st.subheader(model_label)
        if not prompt.strip():
            st.info("Enter a question above to see results.")
            return
        if not res:
            st.info("No result yet. Click Solve (or Run both).")
            return
        thinking = res.get("thinking")
        answer = res.get("answer")
        thinking_expander = st.expander("Thinking Process...", expanded=False)
        if thinking:
            # Use plain text to avoid syntax highlighting / multi-color rendering.
            thinking_expander.text(thinking)
        else:
            thinking_expander.write("(no thinking detected)")
        st.markdown("**Answer**")
        if answer:
            st.success(answer)
        else:
            st.warning("No answer detected.")

    if compare_mode:
        left, right = st.columns(2)
        with left:
            _render_result_box("Base Model")
        with right:
            _render_result_box("CoT Fine-Tuned")
    else:
        _render_result_box(model_choice)


def main() -> None:  # pragma: no cover
    render()


if __name__ == "__main__":  # pragma: no cover
    main()
