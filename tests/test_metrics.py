import types

from reasoning_engine_cot.eval import metrics


def test_strict_adherence():
    t, a = metrics.strict_adherence("<thinking>x</thinking><answer>y</answer>")
    assert t and a
    t, a = metrics.strict_adherence("no tags")
    assert not t and not a


def test_recovered_adherence():
    t, a = metrics.recovered_adherence("some <thinking> but no closing <answer> tag yet")
    assert t and a


def test_tokens_stats():
    class DummyTok:
        def encode(self, txt):
            return list(txt.split())

    tokens, ms_per_token, tps = metrics.tokens_stats(DummyTok(), "a b c", latency_ms=30)
    assert tokens == 3
    assert ms_per_token == 10
    assert tps == 100


def test_peak_memory_handles_no_cuda(monkeypatch):
    fake_cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr(metrics.torch, "cuda", fake_cuda)
    assert metrics.peak_memory_mb() is None


