from datasets import Dataset

from reasoning_engine_cot.data.filters import QualityFilter


def test_quality_filter_enforces_length_and_tags() -> None:
    ds = Dataset.from_dict(
        {
            "text": [
                "<thinking>This is sufficient reasoning text with enough length and additional words to pass filter.</thinking><answer>ok</answer>",
                "missing tags",
            ]
        }
    )
    flt = QualityFilter(min_text_length=20, require_tags=True)
    filtered = flt.filter(ds)
    assert len(filtered) == 1
    assert "<thinking>" in filtered[0]["text"]
