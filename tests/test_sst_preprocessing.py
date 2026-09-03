from sentiment_manifold.data.preprocessing.sst import (
    NEUTRAL_REMOVED_BINARIZATION,
    TIGGES_BINARIZATION,
    apply_labels_to_scored_rows,
    binarize_rows,
    binary_label_without_neutral,
    make_directed_pairs,
    make_maximal_matches,
    remove_neutral,
    tigges_binary_label,
)
from sentiment_manifold.cli import _token_from_environment


class _WhitespaceTokenizer:
    bos_token_id = 99
    init_kwargs = {"_commit_hash": "fake-tokenizer-revision"}

    def __len__(self):
        return 100

    def __call__(self, text, *, add_special_tokens=False):
        ids = list(range(len(text.split())))
        if add_special_tokens:
            ids = [98, *ids]
        return {"input_ids": ids}


def test_publish_requires_explicit_token():
    from sentiment_manifold.data.preprocessing.sst import publish_dataset_configs

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


def test_tigges_binary_collapse_and_named_variants_have_exact_boundaries():
    assert tigges_binary_label(0.0) == 0
    assert tigges_binary_label(0.5) == 0
    assert tigges_binary_label(0.50001) == 1
    assert tigges_binary_label(1.0) == 1
    source = [
        {"example_id": "low", "sentiment_score": 0.4},
        {"example_id": "tie", "sentiment_score": 0.5},
        {"example_id": "upper", "sentiment_score": 0.6},
        {"example_id": "high", "sentiment_score": 0.8},
    ]
    tigges = binarize_rows(source, TIGGES_BINARIZATION)
    assert [row["label"] for row in tigges] == [0, 0, 1, 1]
    assert all(row["binarization_method"] == TIGGES_BINARIZATION for row in tigges)
    neutral_removed = binarize_rows(source, NEUTRAL_REMOVED_BINARIZATION)
    assert [row["example_id"] for row in neutral_removed] == ["low", "high"]
    assert [row["label"] for row in neutral_removed] == [0, 1]


def test_shared_pythia_scores_are_relabelled_and_refiltered_per_variant():
    scored = [
        {
            "example_id": "tie",
            "positive_minus_negative_logit": 2.0,
            "predicted_label": 1,
        }
    ]
    tigges = binarize_rows(
        [{"example_id": "tie", "sentiment_score": 0.5}], TIGGES_BINARIZATION
    )
    rebound = apply_labels_to_scored_rows(scored, tigges)
    assert rebound[0]["label"] == 0
    assert rebound[0]["signed_correct_logit_diff"] == -2.0
    assert rebound[0]["pythia_correct"] is False


def test_preprocess_saves_complete_config_family_for_both_methods(tmp_path, monkeypatch):
    import sentiment_manifold.data.preprocessing.sst as module

    source_rows = []
    for index, score in enumerate((0.4, 0.5, 0.6, 0.8), start=1):
        source_rows.append(
            {
                "example_id": f"sst-{index}",
                "sentence_index": index,
                "phrase_id": index + 100,
                "text": f"review-{index}",
                "sentiment_score": score,
                "original_label_5": "neutral",
                "label": binary_label_without_neutral(score),
                "label_name": None,
                "split": "test",
                "is_neutral_removed": binary_label_without_neutral(score) is None,
            }
        )

    scoring_calls = []

    def fake_score(rows, **kwargs):
        scoring_calls.append(kwargs)
        scored = []
        for row in rows:
            predicted = int(row["sentiment_score"] > 0.5)
            diff = 1.0 if predicted else -1.0
            scored.append(
                {
                    **row,
                    "prompt": f"Review Text: {row['text']} Review Sentiment:",
                    "pythia_raw_num_tokens": 4,
                    "pythia_prompt_num_tokens": 9,
                    "positive_minus_negative_logit": diff,
                    "predicted_label": predicted,
                    "signed_correct_logit_diff": 1.0,
                    "pythia_correct": predicted == row["label"],
                }
            )
        return scored, {
            "filter_model_alias": kwargs["model_name"],
            "filter_model_name": "EleutherAI/pythia-2.8b",
        }

    monkeypatch.setattr(module, "load_sst_source_sentences", lambda _root: source_rows)
    monkeypatch.setattr(module, "score_test_sentences_with_pythia", fake_score)
    result = module.preprocess_sst(
        sst_root=tmp_path,
        output_dir=tmp_path / "processed",
        pairing_models=["gpt2-small"],
        tokenizers={"gpt2-small": _WhitespaceTokenizer()},
    )
    expected = {
        f"{method}_{stage}"
        for method in (TIGGES_BINARIZATION, NEUTRAL_REMOVED_BINARIZATION)
        for stage in (
            "binarized",
            "pythia_scored",
            "pythia_correct",
            "matched_pairs",
            "directed_pairs",
        )
    }
    rq2_expected = {
        f"{method}_{stage}"
        for method in (TIGGES_BINARIZATION, NEUTRAL_REMOVED_BINARIZATION)
        for stage in (
            "pairing_candidates",
            "gpt2_small_matched_pairs",
            "gpt2_small_directed_pairs",
            "common_matched_pairs",
            "common_directed_pairs",
        )
    }
    assert set(result.datasets) == expected | rq2_expected
    assert set(result.metadata["dataset_configs"]) == expected | rq2_expected
    assert all((result.output_dir / name).is_dir() for name in expected | rq2_expected)
    assert result.metadata["pairing_models"] == ["gpt2-small"]
    assert scoring_calls[0]["model_name"] == "pythia-2.8b"


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
