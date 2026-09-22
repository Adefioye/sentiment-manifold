"""Deterministic AIT train/eval/test materialization from published matched pairs."""

from __future__ import annotations

import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...datasets import CounterfactualPair, HuggingFaceRows, TextExample, load_hf_parquet_rows
from .config import AITValenceExperimentConfig

RowsLoader = Callable[..., HuggingFaceRows]


@dataclass(frozen=True)
class PreparedAITData:
    train_examples: tuple[TextExample, ...]
    train_pairs: tuple[CounterfactualPair, ...]
    eval_pairs: tuple[CounterfactualPair, ...]
    test_pairs: tuple[CounterfactualPair, ...]
    sample_manifest: tuple[dict[str, Any], ...]
    pair_manifest: tuple[dict[str, Any], ...]
    requested_revision: str | None
    resolved_revision: str


def _token_from_environment(variable_name: str) -> str | None:
    token = os.environ.get(variable_name)
    if token:
        return token
    token_path = os.environ.get("HF_TOKEN_PATH")
    if not token_path:
        return None
    path = Path(token_path).expanduser()
    if not path.is_file():
        raise ValueError(f"HF_TOKEN_PATH does not point to a readable file: {path}")
    return path.read_text(encoding="utf-8").strip() or None


def _validate_match(row: Mapping[str, Any], *, split: str) -> None:
    required = {
        "pair_id",
        "positive_example_id",
        "positive_prompt",
        "negative_example_id",
        "negative_prompt",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"AIT matched-pair row is missing columns: {missing}")
    row_split = str(row.get("split", split))
    if row_split != split:
        raise ValueError(f"AIT row {row['pair_id']!r} belongs to {row_split!r}, not {split!r}")


def _example(
    row: Mapping[str, Any],
    polarity: str,
    *,
    split: str,
    role: str,
    repo_id: str,
    config_name: str,
    revision: str,
) -> TextExample:
    label = int(polarity == "positive")
    score = row.get(f"{polarity}_source_score")
    return TextExample(
        text=str(row[f"{polarity}_prompt"]),
        label=label,
        example_id=str(row[f"{polarity}_example_id"]),
        metadata={
            "dataset": "ait_valence",
            "role": role,
            "split": split,
            "pair_id": str(row["pair_id"]),
            "source_text": row.get(f"{polarity}_text"),
            "original_valence_class": score,
            "dataset_repo_id": repo_id,
            "dataset_config": config_name,
            "resolved_dataset_revision": revision,
        },
    )


def _directed_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    role: str,
    repo_id: str,
    config_name: str,
    revision: str,
) -> tuple[CounterfactualPair, ...]:
    pairs: list[CounterfactualPair] = []
    for row in rows:
        positive = _example(
            row,
            "positive",
            split=split,
            role=role,
            repo_id=repo_id,
            config_name=config_name,
            revision=revision,
        )
        negative = _example(
            row,
            "negative",
            split=split,
            role=role,
            repo_id=repo_id,
            config_name=config_name,
            revision=revision,
        )
        for source, target, direction in (
            (positive, negative, "negative_to_positive"),
            (negative, positive, "positive_to_negative"),
        ):
            case_id = f"{row['pair_id']}-{direction}"
            common = {"case_id": case_id, "direction": direction}
            pairs.append(
                CounterfactualPair(
                    clean=TextExample(
                        text=source.text,
                        label=source.label,
                        example_id=source.example_id,
                        metadata={**source.metadata, **common, "pair_role": "activation_donor"},
                    ),
                    corrupted=TextExample(
                        text=target.text,
                        label=target.label,
                        example_id=target.example_id,
                        metadata={**target.metadata, **common, "pair_role": "receiver_baseline"},
                    ),
                )
            )
    return tuple(pairs)


def _sample_matches(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    split: str,
    seed: int,
) -> tuple[Mapping[str, Any], ...]:
    ordered = sorted(rows, key=lambda row: str(row["pair_id"]))
    if len(ordered) < count:
        raise RuntimeError(
            f"AIT {split!r} has {len(ordered)} common matches, but {count} are required"
        )
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return tuple(ordered[:count])


def _sample_rows(
    loaded: HuggingFaceRows,
    *,
    split: str,
    count: int,
    seed: int,
) -> tuple[Mapping[str, Any], ...]:
    for row in loaded.rows:
        _validate_match(row, split=split)
    return _sample_matches(loaded.rows, count=count, split=split, seed=seed)


class AITDatasetLoader:
    """Load one immutable AIT revision and assign disjoint experimental roles."""

    def __init__(
        self,
        config: AITValenceExperimentConfig,
        *,
        rows_loader: RowsLoader = load_hf_parquet_rows,
    ) -> None:
        self.config = config
        self.rows_loader = rows_loader

    def load(self) -> PreparedAITData:
        data = self.config.data
        sampling = self.config.sampling
        token = _token_from_environment(data.hf_token_env)
        loaded_by_role = {
            role: self.rows_loader(
                data.repo_id,
                config_name=data.matched_config,
                split=split,
                revision=data.revision,
                token=token,
            )
            for role, split in (
                ("train", data.train_split),
                ("eval", data.eval_split),
                ("test", data.test_split),
            )
        }
        resolved = {loaded.resolved_revision for loaded in loaded_by_role.values()}
        if len(resolved) != 1:
            raise RuntimeError(f"AIT splits resolved to different revisions: {resolved}")
        resolved_revision = next(iter(resolved))

        train_match_count = (sampling.train_examples + 1) // 2
        selected_train = _sample_rows(
            loaded_by_role["train"],
            split=data.train_split,
            count=train_match_count,
            seed=self.config.seed,
        )
        complete_train_matches = selected_train[: sampling.train_examples // 2]
        train_pairs = _directed_pairs(
            complete_train_matches,
            split=data.train_split,
            role="train",
            repo_id=data.repo_id,
            config_name=data.matched_config,
            revision=resolved_revision,
        )
        train_examples: list[TextExample] = []
        for row in complete_train_matches:
            train_examples.extend(
                [
                    _example(
                        row,
                        polarity,
                        split=data.train_split,
                        role="train",
                        repo_id=data.repo_id,
                        config_name=data.matched_config,
                        revision=resolved_revision,
                    )
                    for polarity in ("positive", "negative")
                ]
            )
        if sampling.train_examples % 2:
            train_examples.append(
                _example(
                    selected_train[-1],
                    "positive",
                    split=data.train_split,
                    role="train",
                    repo_id=data.repo_id,
                    config_name=data.matched_config,
                    revision=resolved_revision,
                )
            )

        selected_eval = _sample_rows(
            loaded_by_role["eval"],
            split=data.eval_split,
            count=sampling.eval_directed_cases // 2,
            seed=self.config.seed + 1,
        )
        selected_test = _sample_rows(
            loaded_by_role["test"],
            split=data.test_split,
            count=sampling.test_directed_cases // 2,
            seed=self.config.seed + 2,
        )
        eval_pairs = _directed_pairs(
            selected_eval,
            split=data.eval_split,
            role="das_checkpoint_validation",
            repo_id=data.repo_id,
            config_name=data.matched_config,
            revision=resolved_revision,
        )
        test_pairs = _directed_pairs(
            selected_test,
            split=data.test_split,
            role="layer_selection",
            repo_id=data.repo_id,
            config_name=data.matched_config,
            revision=resolved_revision,
        )

        role_ids = {
            "train": {example.example_id for example in train_examples},
            "eval": {
                example.example_id
                for pair in eval_pairs
                for example in (pair.clean, pair.corrupted)
            },
            "test": {
                example.example_id
                for pair in test_pairs
                for example in (pair.clean, pair.corrupted)
            },
        }
        for left, right in (("train", "eval"), ("train", "test"), ("eval", "test")):
            overlap = role_ids[left] & role_ids[right]
            if overlap:
                raise RuntimeError(f"AIT {left}/{right} leakage: {sorted(overlap)}")

        das_train_ids = {
            example.example_id
            for pair in train_pairs
            for example in (pair.clean, pair.corrupted)
        }
        sample_rows = [
            {
                "role": "train",
                "source_split": data.train_split,
                "example_id": example.example_id,
                "label": example.label,
                "original_valence_class": example.metadata.get("original_valence_class"),
                "used_by_linear_training": True,
                "used_by_das_training": example.example_id in das_train_ids,
                "used_for_das_checkpoint_validation": False,
                "used_for_layer_selection": False,
                "prompt": example.text,
            }
            for example in train_examples
        ]
        for role, pairs in (("eval", eval_pairs), ("test", test_pairs)):
            examples_by_id = {
                example.example_id: example
                for pair in pairs
                for example in (pair.clean, pair.corrupted)
            }
            sample_rows.extend(
                {
                    "role": role,
                    "source_split": example.metadata["split"],
                    "example_id": example.example_id,
                    "label": example.label,
                    "original_valence_class": example.metadata.get(
                        "original_valence_class"
                    ),
                    "used_by_linear_training": False,
                    "used_by_das_training": False,
                    "used_for_das_checkpoint_validation": role == "eval",
                    "used_for_layer_selection": role == "test",
                    "prompt": example.text,
                }
                for example in sorted(examples_by_id.values(), key=lambda row: row.example_id)
            )
        sample_manifest = tuple(sample_rows)
        pair_manifest = tuple(
            {
                "role": role,
                "source_split": pair.clean.metadata["split"],
                "case_id": pair.clean.metadata["case_id"],
                "pair_id": pair.clean.metadata["pair_id"],
                "direction": pair.clean.metadata["direction"],
                "source_example_id": pair.clean.example_id,
                "source_label": pair.clean.label,
                "target_example_id": pair.corrupted.example_id,
                "target_label": pair.corrupted.label,
            }
            for role, pairs in (
                ("train", train_pairs),
                ("eval", eval_pairs),
                ("test", test_pairs),
            )
            for pair in pairs
        )
        return PreparedAITData(
            train_examples=tuple(train_examples),
            train_pairs=train_pairs,
            eval_pairs=eval_pairs,
            test_pairs=test_pairs,
            sample_manifest=sample_manifest,
            pair_manifest=pair_manifest,
            requested_revision=data.revision,
            resolved_revision=resolved_revision,
        )


__all__ = ["AITDatasetLoader", "PreparedAITData"]
