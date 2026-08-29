from sentiment_manifold.data.sst_preprocessing import (
    binary_label_without_neutral,
    make_directed_pairs,
    make_maximal_matches,
    remove_neutral,
)
from sentiment_manifold.cli import _token_from_environment


def test_publish_requires_explicit_token():
    from sentiment_manifold.data.sst_preprocessing import publish_dataset_configs

    try:
        publish_dataset_configs({}, repo_id="example/repo", private=True, card_text="", token="")
    except ValueError as error:
        assert "token is required" in str(error)
    else:
        raise AssertionError("Publishing without an explicit token should fail")


def test_token_can_come_from_named_environment_or_token_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_HF_TOKEN", "environment-secret")
    assert _token_from_environment("PROJECT_HF_TOKEN") == "environment-secret"

    monkeypatch.delenv("PROJECT_HF_TOKEN")
    token_path = tmp_path / "hf-token"
    token_path.write_text("file-secret\n")
    monkeypatch.setenv("HF_TOKEN_PATH", str(token_path))
    assert _token_from_environment("PROJECT_HF_TOKEN") == "file-secret"


def _row(index: int, label: int, raw_length: int, prompt_length: int, *, correct: bool = True):
    polarity = "positive" if label else "negative"
    return {
        "example_id": f"sst-{index}",
        "sentence_index": index,
        "phrase_id": index + 100,
        "text": f"{polarity}-{index}",
        "sentiment_score": 0.8 if label else 0.2,
        "label": label,
        "label_name": polarity,
        "split": "test",
        "prompt": f"Review Text: {polarity}-{index} Review Sentiment:",
        "pythia_raw_num_tokens": raw_length,
        "pythia_prompt_num_tokens": prompt_length,
        "signed_correct_logit_diff": 1.0,
        "pythia_correct": correct,
    }


def test_binary_collapse_removes_official_neutral_interval():
    assert binary_label_without_neutral(0.0) == 0
    assert binary_label_without_neutral(0.4) == 0
    assert binary_label_without_neutral(0.40001) is None
    assert binary_label_without_neutral(0.6) is None
    assert binary_label_without_neutral(0.60001) == 1
    assert binary_label_without_neutral(1.0) == 1
    rows = [{"label": 0}, {"label": None}, {"label": 1}]
    assert [row["label"] for row in remove_neutral(rows)] == [0, 1]


def test_matches_are_maximal_non_reused_opposite_and_equal_length():
    rows = [
        _row(1, 1, 4, 9),
        _row(2, 1, 4, 9),
        _row(3, 0, 4, 9),
        _row(4, 0, 5, 10),
        _row(5, 1, 5, 10, correct=False),
    ]
    matches = make_maximal_matches(rows)
    assert len(matches) == 1
    match = matches[0]
    assert match["positive_example_id"] == "sst-1"
    assert match["negative_example_id"] == "sst-3"
    assert match["pythia_raw_num_tokens"] == 4
    assert match["pythia_prompt_num_tokens"] == 9

    directed = make_directed_pairs(matches)
    assert len(directed) == 2
    assert {row["clean_label"] for row in directed} == {0, 1}
    assert all(row["clean_label"] != row["corrupted_label"] for row in directed)
    assert {row["direction"] for row in directed} == {
        "positive_to_negative",
        "negative_to_positive",
    }
