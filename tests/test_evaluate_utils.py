from scripts.evaluate import answer_contains, compute_adherence


def test_compute_adherence_detects_both_tags():
    txt = "<thinking>reason</thinking> ... <answer>42</answer>"
    thinking_ok, answer_ok = compute_adherence(txt)
    assert thinking_ok is True
    assert answer_ok is True


def test_compute_adherence_handles_missing_tags():
    txt = "no tags here"
    thinking_ok, answer_ok = compute_adherence(txt)
    assert thinking_ok is False
    assert answer_ok is False


def test_answer_contains_case_insensitive():
    assert answer_contains("The answer is Egg.", "egg")
    assert answer_contains("2 apples remain", "2")
    assert not answer_contains("nothing relevant", "egg")




