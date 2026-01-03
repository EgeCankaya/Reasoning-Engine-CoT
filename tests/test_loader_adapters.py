import types
from pathlib import Path

from reasoning_engine_cot.inference.loader import ModelLoader


def test_loader_uses_base_then_applies_adapters(monkeypatch, tmp_path):
    calls = {"base_path": None, "adapter_path": None}

    # Fake tokenizer/model objects
    class DummyModel:
        adapter_loaded = None

    dummy_model = DummyModel()
    dummy_tokenizer = object()

    # Fake FastLanguageModel
    fake_fast = types.SimpleNamespace()

    def fake_from_pretrained(path, *_, **__):
        calls["base_path"] = path
        return dummy_model, dummy_tokenizer

    fake_fast.from_pretrained = fake_from_pretrained

    # Fake PeftModel
    def fake_peft_from_pretrained(model, adapter_dir):
        calls["adapter_path"] = str(adapter_dir)
        model.adapter_loaded = str(adapter_dir)
        return model

    fake_peft = types.SimpleNamespace(from_pretrained=fake_peft_from_pretrained)

    monkeypatch.setitem(
        __import__("sys").modules, "unsloth", types.SimpleNamespace(FastLanguageModel=fake_fast)
    )
    # loader imports `from peft import PeftModel`
    monkeypatch.setitem(__import__("sys").modules, "peft", types.SimpleNamespace(PeftModel=fake_peft))

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()

    loader = ModelLoader(model_name="models/base", adapter_path=str(adapter_dir))
    model, tokenizer = loader.load()

    assert calls["base_path"] == "models/base"
    assert calls["adapter_path"] == str(adapter_dir)
    assert getattr(model, "adapter_loaded") == str(adapter_dir)
    assert tokenizer is dummy_tokenizer




