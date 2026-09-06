"""Build auditable tabular manifests for sentiment-position experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from ...datasets.toy_movie_review import ToyEvaluationSet, ToyMovieReview
from ...datasets.types import CounterfactualPair, TextExample
from ...models import CausalLMAdapter

AnswerSpec = Mapping[int, Sequence[str]]


def answer_token_rows(
    adapter: CausalLMAdapter,
    answers_by_dataset: Mapping[str, AnswerSpec],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, answers in answers_by_dataset.items():
        for label, values in answers.items():
            for pair_index, answer in enumerate(values):
                rows.append(
                    {
                        "dataset": dataset,
                        "label": label,
                        "pair_index": pair_index,
                        "answer": answer,
                        "answer_repr": repr(answer),
                        "token_id": adapter.single_token_id(answer),
                    }
                )
    return rows


def prompt_rows(
    adapter: CausalLMAdapter,
    dataset: str,
    split: str,
    examples: Sequence[TextExample],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        batch = adapter.tokenize([example])
        mask = batch.attention_mask[0].bool()
        token_ids = batch.input_ids[0][mask].tolist()
        focus_position = None if batch.focus_positions is None else int(batch.focus_positions[0])
        if focus_position is not None and focus_position < 0:
            focus_position = None
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "example_id": example.example_id,
                "label": example.label,
                "prompt_text": example.text,
                "prompt_sha256": hashlib.sha256(example.text.encode("utf-8")).hexdigest(),
                "token_ids": json.dumps(token_ids),
                "num_tokens": len(token_ids),
                "focus_position": focus_position,
                "focus_word": example.metadata.get("focus_word")
                or example.metadata.get("adjective"),
                "focus_word_type": example.metadata.get("focus_word_type", "adjective"),
            }
        )
    return rows


def pair_rows(
    adapter: CausalLMAdapter,
    dataset: str,
    pairs: Sequence[CounterfactualPair],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        clean = adapter.tokenize([pair.clean])
        corrupted = adapter.tokenize([pair.corrupted])
        clean_length = int(clean.attention_mask.sum())
        corrupted_length = int(corrupted.attention_mask.sum())
        rows.append(
            {
                "dataset": dataset,
                "case_index": index,
                "case_id": pair.clean.metadata.get("case_id", f"{dataset}-{index:05d}"),
                "pair_id": pair.clean.metadata.get("pair_id"),
                "direction": pair.clean.metadata.get(
                    "direction", f"{pair.corrupted.label}_to_{pair.clean.label}"
                ),
                "clean_id": pair.clean.example_id,
                "corrupted_id": pair.corrupted.example_id,
                "clean_label": pair.clean.label,
                "corrupted_label": pair.corrupted.label,
                "clean_text": pair.clean.text,
                "corrupted_text": pair.corrupted.text,
                "clean_num_tokens": clean_length,
                "corrupted_num_tokens": corrupted_length,
                "equal_token_length": clean_length == corrupted_length,
            }
        )
    return rows


def vocabulary_rows(
    adapter: CausalLMAdapter,
    raw_toy: ToyMovieReview,
    filtered_toy: ToyMovieReview,
    evaluations: Mapping[str, ToyEvaluationSet],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "test"):
        for label in (1, 0):
            retained = set(filtered_toy.adjectives[split][label])
            for word in raw_toy.adjectives[split][label]:
                token_ids = adapter.tokenizer(" " + word.strip(), add_special_tokens=False)[
                    "input_ids"
                ]
                rows.append(
                    {
                        "dataset": "toy_train" if split == "train" else "toy_adjectives",
                        "split": split,
                        "label": label,
                        "word_type": "adjective",
                        "word": word,
                        "retained": word in retained,
                        "token_ids": json.dumps(token_ids),
                        "num_tokens": len(token_ids),
                    }
                )
    rows.extend(_lexical_evaluation_rows(adapter, raw_toy, evaluations))
    return rows


def _lexical_evaluation_rows(
    adapter: CausalLMAdapter,
    raw_toy: ToyMovieReview,
    evaluations: Mapping[str, ToyEvaluationSet],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = (
        ("toy_verbs", "verb", raw_toy.verbs),
        ("toy_adverbs", "adverb", raw_toy.adverbs),
    )
    for dataset, word_type, words_by_label in sources:
        retained = {example.metadata["focus_word"] for example in evaluations[dataset].examples}
        for label in (1, 0):
            seen: set[str] = set()
            for word in words_by_label[label]:
                token_ids = adapter.tokenizer(" " + word.strip(), add_special_tokens=False)[
                    "input_ids"
                ]
                row = {
                    "dataset": dataset,
                    "split": "test",
                    "label": label,
                    "word_type": word_type,
                    "word": word,
                    "retained": word in retained and word not in seen,
                    "token_ids": json.dumps(token_ids),
                    "num_tokens": len(token_ids),
                }
                if word_type == "adverb":
                    row["duplicate"] = word in seen
                rows.append(row)
                seen.add(word)
    return rows


__all__ = ["answer_token_rows", "pair_rows", "prompt_rows", "vocabulary_rows"]
