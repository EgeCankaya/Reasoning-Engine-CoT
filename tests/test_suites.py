import eval.suites as suites


def test_load_riddles(tmp_path):
    path = tmp_path / "riddles.jsonl"
    path.write_text('{"id":"r1","question":"Q?","answer":"A"}\n', encoding="utf-8")
    items = suites.load_riddles(path)
    assert len(items) == 1
    assert items[0].question == "Q?"


def test_load_gsm8k_lite_mock(monkeypatch):
    class DummyRows:
        def __init__(self, rows):
            self._rows = rows

        def select(self, rng):
            return self._rows[: len(rng)]

        def __len__(self):
            return len(self._rows)

    dummy = DummyRows([
        {"question": "1+1?", "answer": "#### 2"},
        {"question": "2+2?", "answer": "4"},
    ])

    def fake_load_dataset(name, config, split):
        return dummy

    monkeypatch.setattr(suites, "load_dataset", fake_load_dataset)
    items = suites.load_gsm8k_lite(limit=1)
    assert len(items) == 1
    assert items[0].answer == "2"




