from types import SimpleNamespace

import pytest
import torch

from sentiment_manifold.models.sentiment_filter import (
    score_binary_rows_with_pythia,
)


class _FilterTokenizer:
    bos_token_id = 0
    eos_token_id = 0
    eos_token = "<eos>"
    pad_token_id = None
    model_max_length = 9
    init_kwargs = {"_commit_hash": "tokenizer-revision"}

    def __call__(self, text, *, add_special_tokens=False):
        if text == " Positive":
            return {"input_ids": [1]}
        if text == " Negative":
            return {"input_ids": [2]}
        ids = []
        for token in text.split():
            if token == "good":
                ids.append(3)
            elif token == "bad":
                ids.append(4)
            else:
                ids.append(5)
        return {"input_ids": ids}

    def pad(self, encoded, **_kwargs):
        rows = encoded["input_ids"]
        width = max(map(len, rows))
        padded = [[0] * (width - len(row)) + row for row in rows]
        masks = [[0] * (width - len(row)) + [1] * len(row) for row in rows]
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class _FilterModel:
    config = SimpleNamespace(max_position_embeddings=9, _commit_hash="model-revision")

    def to(self, _device):
        return self

    def eval(self):
        return self

    def requires_grad_(self, _requires_grad):
        return self

    def __call__(self, *, input_ids, attention_mask, use_cache):
        del attention_mask, use_cache
        logits = torch.zeros((*input_ids.shape, 6), dtype=torch.float32)
        for index, row in enumerate(input_ids):
            if bool((row == 3).any()):
                logits[index, -1, 1] = 2.0
            if bool((row == 4).any()):
                logits[index, -1, 2] = 2.0
        return SimpleNamespace(logits=logits)


def test_shared_filter_scores_gold_labels_and_skips_over_context_rows(monkeypatch):
    import sentiment_manifold.models.sentiment_filter as module

    tokenizer = _FilterTokenizer()
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: _FilterModel(),
    )
    rows = [
        {"example_id": "positive", "text": "good", "label": 1, "split": "test"},
        {"example_id": "negative", "text": "bad", "label": 0, "split": "test"},
        {"example_id": "wrong", "text": "good", "label": 0, "split": "test"},
        {
            "example_id": "too-long",
            "text": "good one two three four five six",
            "label": 1,
            "split": "test",
        },
        {"example_id": "train", "text": "good", "label": 1, "split": "train"},
    ]

    scored, metadata = score_binary_rows_with_pythia(
        rows,
        model_name="pythia-2.8b",
        device="cpu",
        dtype="float32",
        batch_size=2,
    )

    assert [row["example_id"] for row in scored] == ["positive", "negative", "wrong"]
    assert [row["pythia_correct"] for row in scored] == [True, True, False]
    assert all(row["filter_model_alias"] == "pythia-2.8b" for row in scored)
    assert metadata["filter_input_rows"] == 4
    assert metadata["filter_scored_rows"] == 3
    assert metadata["filter_over_context_rows"] == 1
    assert metadata["resolved_filter_model_revision"] == "model-revision"


def test_shared_filter_rejects_non_binary_labels(monkeypatch):
    import sentiment_manifold.models.sentiment_filter as module

    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: _FilterTokenizer(),
    )
    monkeypatch.setattr(
        module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: _FilterModel(),
    )
    with pytest.raises(ValueError, match="labels 0/1"):
        score_binary_rows_with_pythia(
            [{"example_id": "neutral", "text": "okay", "label": 2, "split": "test"}],
            device="cpu",
        )
