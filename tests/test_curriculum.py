from datasets import Dataset

from reasoning_engine_cot.data.curriculum import CurriculumSorter


def test_curriculum_sort_orders_by_reasoning_length() -> None:
    ds = Dataset.from_dict(
        {
            "text": [
                "<thinking>short steps</thinking><answer>1</answer>",
                "<thinking>this has a lot more words in the reasoning block for testing ordering</thinking><answer>2</answer>",
            ]
        }
    )
    sorter = CurriculumSorter()
    sorted_ds = sorter.sort(ds)
    assert sorted_ds[0]["text"].startswith("<thinking>short")
