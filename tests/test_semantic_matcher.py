import numpy as np

from reasoning_engine_cot.eval.metrics import ReasoningCoherenceScorer, SemanticMatcher


class _FakeModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):  # noqa: ANN001
        return np.array([[1.0, 0.0], [1.0, 0.0]])


def test_semantic_matcher_similarity_uses_model() -> None:
    matcher = SemanticMatcher()
    matcher.model = _FakeModel()
    score = matcher.similarity("a", "b")
    assert score == 1.0


def test_reasoning_coherence_scores_steps_and_answer() -> None:
    scorer = ReasoningCoherenceScorer()
    thinking = "- step1\n- step2 includes answer: 42\n- step3"
    answer = "42"
    score = scorer.score(thinking, answer)
    assert score > 0.3
