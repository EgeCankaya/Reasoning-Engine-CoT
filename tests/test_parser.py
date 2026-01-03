from reasoning_engine_cot.inference.generator import ReasoningGenerator


def test_parse_response_extracts_sections():
    sample = (
        "<thinking>Initial count is 3. Action is eating 1. 3 minus 1 equals 2.</thinking>"
        "<answer>2 apples</answer>"
    )
    parsed = ReasoningGenerator.parse_response(sample)
    assert parsed["thinking"].startswith("Initial count")
    assert parsed["answer"] == "2 apples"


def test_parse_response_handles_missing_tags():
    parsed = ReasoningGenerator.parse_response("No tags here")
    assert parsed["thinking"] is None
    assert parsed["answer"] is None


def test_extract_partial_sections_strips_control_tokens_and_marker_only_output():
    # Emulates Qwen-style verbose control tokens that can appear when prompts are malformed.
    junk = (
        "<|initial_analysis_id|>initial_analysis <|conscious_thought_id|>conscious_thought "
        "<|self_improvement_id|>self_improvement <|reflection_id|>reflection <|feeling_id|>feeling"
    )
    thinking, answer = ReasoningGenerator._extract_partial_sections(junk)  # type: ignore[attr-defined]
    assert thinking is None
    assert answer is None


def test_extract_partial_sections_does_not_drop_normal_letters():
    sample = "<thinking>I need to think about the structure of the statement.</thinking><answer>OK</answer>"
    thinking, answer = ReasoningGenerator._extract_partial_sections(sample)  # type: ignore[attr-defined]
    assert thinking is not None
    assert "to" in thinking
    assert "the" in thinking
    assert answer == "OK"


def test_extract_partial_sections_does_not_return_confidence_as_answer_when_no_answer_tag():
    # If a model emits verbose "metadata" after thinking, we should not treat it as the answer.
    sample = (
        "<thinking>Reasoning here.</thinking>\n"
        "Palindrome\n"
        "Confidence: High\n"
        "Emotional response: Intrigued\n"
    )
    thinking, answer = ReasoningGenerator._extract_partial_sections(sample)  # type: ignore[attr-defined]
    assert thinking == "Reasoning here."
    assert answer == "Palindrome"




















