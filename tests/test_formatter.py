from datasets import Dataset

from reasoning_engine_cot.data import CoTFormatter


def test_format_sample_includes_tags():
    formatter = CoTFormatter()
    text = formatter.format_sample("What is 2+2?", "Add 2 and 2", "4")
    assert "<thinking>" in text and "</thinking>" in text
    assert "<answer>" in text and "</answer>" in text


def test_format_dataset_adds_text_column():
    data = Dataset.from_list([
        {"question": "Q1", "reasoning": "R1", "final_answer": "A1"},
        {"question": "Q2", "reasoning": "R2", "final_answer": "A2"},
    ])
    formatted = CoTFormatter().format_dataset(data)
    assert "text" in formatted.column_names
    sample_text = formatted[0]["text"]
    assert "Q1" in sample_text and "R1" in sample_text and "A1" in sample_text




