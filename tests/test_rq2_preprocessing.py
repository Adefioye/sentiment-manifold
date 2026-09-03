import json

from sentiment_manifold.data.preprocessing.ait import load_ait_binary, preprocess_ait
from sentiment_manifold.data.preprocessing.common import (
    PairingModelSpec,
    annotate_token_lengths,
    make_directed_pairs,
    make_equal_length_matches,
    resolve_pairing_models,
)
from sentiment_manifold.data.preprocessing.dynasent import (
    load_dynasent_binary,
    preprocess_dynasent,
)
from sentiment_manifold.data.preprocessing.imdb import imdb_rows, preprocess_imdb


class _WhitespaceTokenizer:
    bos_token_id = 99
    init_kwargs = {"_commit_hash": "fake-revision"}

    def __init__(self, special_tokens=0):
        self.special_tokens = special_tokens

    def __len__(self):
        return 101

    def __call__(self, text, *, add_special_tokens=False):
        count = len(text.split())
        ids = list(range(count))
        if add_special_tokens:
            ids = list(range(self.special_tokens)) + ids
        return {"input_ids": ids}


def test_all_four_pairing_models_are_available():
    specs = resolve_pairing_models(None)
    assert [spec.alias for spec in specs] == [
        "gpt2-small",
        "qwen-0.6b",
        "gemma-2b",
        "pythia-1.4b",
    ]
    assert [spec.hub_name for spec in specs] == [
        "gpt2",
        "Qwen/Qwen3-0.6B-Base",
        "google/gemma-2b",
        "EleutherAI/pythia-1.4b",
    ]


def test_model_specific_and_common_pairs_have_equal_full_prompt_lengths():
    specs = resolve_pairing_models(["gpt2-small", "qwen-0.6b"])
    rows = [
        {"example_id": "p1", "text": "very good", "label": 1, "split": "test"},
        {"example_id": "n1", "text": "very bad", "label": 0, "split": "test"},
        {"example_id": "p2", "text": "excellent", "label": 1, "split": "test"},
    ]
    annotated, _ = annotate_token_lengths(
        rows,
        specs=specs,
        tokenizers={
            "gpt2-small": _WhitespaceTokenizer(),
            "qwen-0.6b": _WhitespaceTokenizer(special_tokens=2),
        },
    )
    matches = make_equal_length_matches(
        annotated,
        dataset_name="fixture",
        specs=specs,
        pairing_model="common",
        splits=("test",),
    )
    assert len(matches) == 1
    assert matches[0]["positive_example_id"] == "p1"
    assert matches[0]["negative_example_id"] == "n1"
    assert matches[0]["gpt2_small_prompt_num_tokens"] == 7
    assert matches[0]["qwen_0_6b_prompt_num_tokens"] == 8
    directed = make_directed_pairs(matches)
    assert len(directed) == 2
    assert {row["direction"] for row in directed} == {
        "negative_to_positive",
        "positive_to_negative",
    }
    neg_to_pos = next(row for row in directed if row["direction"] == "negative_to_positive")
    assert neg_to_pos["source_label"] == 1
    assert neg_to_pos["target_label"] == 0


def test_over_context_prompts_are_annotated_and_not_paired():
    spec = PairingModelSpec("tiny", "fixture/tiny", "tiny", True, 3)
    rows = [
        {"example_id": "p", "text": "very good", "label": 1, "split": "test"},
        {"example_id": "n", "text": "very bad", "label": 0, "split": "test"},
    ]
    annotated, _ = annotate_token_lengths(
        rows,
        specs=[spec],
        tokenizers={"tiny": _WhitespaceTokenizer()},
    )
    assert all(row["tiny_fits_context"] is False for row in annotated)
    assert make_equal_length_matches(
        annotated,
        dataset_name="fixture",
        specs=[spec],
        pairing_model="tiny",
        splits=("test",),
    ) == []


def _write_ait(path, rows):
    header = "ID\tTweet\tAffect Dimension\tIntensity Class\n"
    body = "".join(f"{row_id}\t{text}\tvalence\t{score}: class\n" for row_id, text, score in rows)
    path.write_text(header + body, encoding="utf-8")


def test_ait_binary_mapping_excludes_zero_and_retains_ordinal_class(tmp_path):
    train = tmp_path / "2018-Valence-oc-En-train.txt"
    _write_ait(train, [("1", "bad tweet", -2), ("2", "neutral tweet", 0), ("3", "good tweet", 3)])
    rows, checksums = load_ait_binary({"train": train})
    assert [row["label"] for row in rows] == [0, 1]
    assert [row["original_valence_class"] for row in rows] == [-2, 3]
    assert len(checksums[str(train.resolve())]) == 64


def test_ait_pipeline_saves_binary_and_pairing_configs(tmp_path):
    files = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"ait-{split}.txt"
        _write_ait(path, [(f"{split}-n", "very bad", -1), (f"{split}-p", "very good", 1)])
        files[split] = path
    result = preprocess_ait(
        output_dir=tmp_path / "processed",
        files=files,
        pairing_models=["gpt2-small"],
        tokenizers={"gpt2-small": _WhitespaceTokenizer()},
    )
    assert set(result.datasets) == {
        "binary",
        "pairing_candidates",
        "gpt2_small_matched_pairs",
        "gpt2_small_directed_pairs",
        "common_matched_pairs",
        "common_directed_pairs",
    }
    assert result.metadata["counts"]["pairing"]["gpt2-small"]["matched_pairs"] == 3
    assert (result.output_dir / "metadata.json").is_file()


def test_imdb_normalization_ignores_unsupervised_rows():
    rows = imdb_rows(
        {
            "train": [{"text": "awful", "label": 0}, {"text": "great", "label": 1}],
            "unsupervised": [{"text": "unknown", "label": -1}],
        }
    )
    assert [row["label"] for row in rows] == [0, 1]
    assert all(row["split"] == "train" for row in rows)
    assert rows[0]["text"] == "awful"


def test_imdb_pipeline_builds_test_pairs_without_network(tmp_path):
    source = {
        "train": [{"text": "training review", "label": 1}],
        "test": [
            {"text": "very bad", "label": 0},
            {"text": "very good", "label": 1},
        ],
    }
    result = preprocess_imdb(
        output_dir=tmp_path / "imdb",
        source_dataset=source,
        pairing_models=["gpt2-small"],
        tokenizers={"gpt2-small": _WhitespaceTokenizer()},
    )
    assert result.metadata["counts"]["pairing"]["gpt2-small"] == {
        "matched_pairs": 1,
        "directed_pairs": 2,
    }


def test_dynasent_keeps_only_gold_binary_labels(tmp_path):
    source = tmp_path / "dynasent-round01-test.jsonl"
    records = [
        {"text_id": "a", "sentence": "great", "gold_label": "positive"},
        {"text_id": "b", "sentence": "bad", "gold_label": "negative"},
        {"text_id": "c", "sentence": "okay", "gold_label": "neutral"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    rows, checksums, removed = load_dynasent_binary({(1, "test"): source})
    assert [row["label"] for row in rows[1]] == [1, 0]
    assert removed == {1: 1}
    assert len(checksums[str(source.resolve())]) == 64


def test_dynasent_pipeline_keeps_rounds_separate(tmp_path):
    files = {}
    for split in ("train", "validation", "test"):
        path = tmp_path / f"dynasent-round02-{split}.jsonl"
        records = [
            {"text_id": f"{split}-p", "sentence": "very good", "gold_label": "positive"},
            {"text_id": f"{split}-n", "sentence": "very bad", "gold_label": "negative"},
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
        )
        files[(2, split)] = path
    result = preprocess_dynasent(
        dynasent_root=tmp_path,
        output_dir=tmp_path / "dynasent",
        rounds=[2],
        files=files,
        pairing_models=["gpt2-small"],
        tokenizers={"gpt2-small": _WhitespaceTokenizer()},
    )
    assert all(name.startswith("r2_") for name in result.datasets)
    assert result.metadata["counts"]["r2"]["pairing"]["gpt2-small"] == {
        "matched_pairs": 1,
        "directed_pairs": 2,
    }
