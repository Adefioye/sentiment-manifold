"""Dataset materialization for fixed-layer selectivity experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from ...datasets import CounterfactualPair, TextExample, load_toy_movie_review
from ...datasets.toy_movie_review import pair_toy_examples
from ...models import CausalLMAdapter
from ...probes import balanced_label_permutation
from ..ait_valence.config import (
    AITDataConfig,
    AITSamplingConfig,
    AITValenceExperimentConfig,
    AITValenceSelectionConfig,
)
from ..ait_valence.datasets import AITDatasetLoader
from .config import FixedLayerModelConfig, FixedLayerSelectivityConfig

ROLE_OFFSETS = {"train": 0, "validation": 1_000_003, "test": 2_000_003}


@dataclass(frozen=True)
class PreparedSelectivityData:
    name: str
    examples: Mapping[str, tuple[TextExample, ...]]
    pairs: Mapping[str, tuple[CounterfactualPair, ...]]
    answers: dict[int, tuple[str, ...]]
    sample_manifest: tuple[dict, ...]
    pair_manifest: tuple[dict, ...]
    provenance: dict[str, str | None]


@dataclass(frozen=True)
class RandomizedSelectivityData:
    examples: Mapping[str, tuple[TextExample, ...]]
    pairs: Mapping[str, tuple[CounterfactualPair, ...]]
    assignment_rows: tuple[dict, ...]


def _stratified_toy_split(
    examples: Sequence[TextExample], *, validation_fraction: float, seed: int
) -> tuple[tuple[TextExample, ...], tuple[TextExample, ...]]:
    training: list[TextExample] = []
    validation: list[TextExample] = []
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        selected = sorted(
            (example for example in examples if example.label == label),
            key=lambda row: row.example_id,
        )
        indices = rng.permutation(len(selected))
        validation_count = max(1, int(round(len(selected) * validation_fraction)))
        validation_indices = set(indices[:validation_count].tolist())
        for index, example in enumerate(selected):
            destination = validation if index in validation_indices else training
            destination.append(example)
    return (
        tuple(sorted(training, key=lambda row: row.example_id)),
        tuple(sorted(validation, key=lambda row: row.example_id)),
    )


def prepare_toy_data(
    config: FixedLayerSelectivityConfig,
    model: FixedLayerModelConfig,
    adapter: CausalLMAdapter,
) -> PreparedSelectivityData:
    raw = load_toy_movie_review(config.data.toy_config)
    filtered = raw.tokenizer_filtered(adapter.tokenizer)
    train, validation = _stratified_toy_split(
        filtered.train,
        validation_fraction=config.data.toy_validation_fraction,
        seed=config.seed,
    )
    examples = {
        "train": train,
        "validation": validation,
        "test": tuple(sorted(filtered.test, key=lambda row: row.example_id)),
    }
    pairs = {
        role: tuple(pair_toy_examples(list(rows))) for role, rows in examples.items()
    }
    sample_manifest = tuple(
        {
            "model": model.name,
            "dataset": "toy_movie_review",
            "role": role,
            "example_id": example.example_id,
            "label": example.label,
            "prompt": example.text,
            "adjective": example.metadata.get("adjective"),
            "verb": example.metadata.get("verb"),
        }
        for role, rows in examples.items()
        for example in rows
    )
    pair_manifest = tuple(
        {
            "model": model.name,
            "dataset": "toy_movie_review",
            "role": role,
            "source_example_id": pair.clean.example_id,
            "source_label": pair.clean.label,
            "target_example_id": pair.corrupted.example_id,
            "target_label": pair.corrupted.label,
        }
        for role, rows in pairs.items()
        for pair in rows
    )
    return PreparedSelectivityData(
        name="toy_movie_review",
        examples=examples,
        pairs=pairs,
        answers=filtered.answers,
        sample_manifest=sample_manifest,
        pair_manifest=pair_manifest,
        provenance={"toy_config": str(config.data.toy_config)},
    )


def _unique_pair_examples(pairs: Sequence[CounterfactualPair]) -> tuple[TextExample, ...]:
    lookup = {
        example.example_id: example
        for pair in pairs
        for example in (pair.clean, pair.corrupted)
    }
    return tuple(sorted(lookup.values(), key=lambda row: row.example_id))


def prepare_ait_data(
    config: FixedLayerSelectivityConfig,
    model: FixedLayerModelConfig,
) -> PreparedSelectivityData:
    ait_config = AITValenceExperimentConfig(
        seed=config.seed,
        models=[model],
        data=AITDataConfig(
            repo_id=config.data.ait_repo_id,
            revision=config.data.ait_revision,
            model_matched_configs=config.data.ait_model_matched_configs,
            train_split=config.data.ait_train_split,
            eval_split=config.data.ait_validation_split,
            test_split=config.data.ait_test_split,
            hf_token_env=config.data.hf_token_env,
            positive_answers=config.data.positive_answers,
            negative_answers=config.data.negative_answers,
        ),
        sampling=AITSamplingConfig(
            train_examples=None,
            eval_directed_cases=None,
            test_directed_cases=None,
        ),
        selection=AITValenceSelectionConfig(
            layer_selection_split="eval",
            final_evaluation_split="test",
        ),
    )
    loaded = AITDatasetLoader(ait_config).load(model.name)
    examples = {
        "train": tuple(sorted(loaded.train_examples, key=lambda row: row.example_id)),
        "validation": _unique_pair_examples(loaded.eval_pairs),
        "test": _unique_pair_examples(loaded.test_pairs),
    }
    pairs = {
        "train": loaded.train_pairs,
        "validation": loaded.eval_pairs,
        "test": loaded.test_pairs,
    }
    return PreparedSelectivityData(
        name="full_ait",
        examples=examples,
        pairs=pairs,
        answers=ait_config.data.answers,
        sample_manifest=tuple(
            {**row, "dataset": "full_ait", "role": "validation" if row["role"] == "eval" else row["role"]}
            for row in loaded.sample_manifest
        ),
        pair_manifest=tuple(
            {**row, "dataset": "full_ait", "role": "validation" if row["role"] == "eval" else row["role"]}
            for row in loaded.pair_manifest
        ),
        provenance={
            "dataset_repo_id": config.data.ait_repo_id,
            "requested_revision": config.data.ait_revision,
            "resolved_revision": loaded.resolved_revision,
            "dataset_config": loaded.dataset_config,
        },
    )


def _opposite_label_pairs(
    examples: Sequence[TextExample], *, seed: int
) -> tuple[CounterfactualPair, ...]:
    rng = np.random.default_rng(seed)
    negative = [example for example in examples if example.label == 0]
    positive = [example for example in examples if example.label == 1]
    rng.shuffle(negative)
    rng.shuffle(positive)
    count = min(len(negative), len(positive))
    if count < 1:
        raise ValueError("Random-label DAS requires both classes in every split")
    pairs: list[CounterfactualPair] = []
    for low, high in zip(negative[:count], positive[:count]):
        pairs.append(CounterfactualPair(clean=high, corrupted=low))
        pairs.append(CounterfactualPair(clean=low, corrupted=high))
    return tuple(pairs)


def randomize_data(
    prepared: PreparedSelectivityData, *, seed: int
) -> RandomizedSelectivityData:
    randomized: dict[str, tuple[TextExample, ...]] = {}
    assignments: list[dict] = []
    pairs: dict[str, tuple[CounterfactualPair, ...]] = {}
    for role, rows in prepared.examples.items():
        original = np.asarray([row.label for row in rows], dtype=np.int64)
        assigned = balanced_label_permutation(original, seed=seed + ROLE_OFFSETS[role])
        relabeled = tuple(
            replace(
                example,
                label=int(label),
                metadata={
                    **example.metadata,
                    "real_label": example.label,
                    "random_label_seed": seed,
                },
            )
            for example, label in zip(rows, assigned)
        )
        randomized[role] = relabeled
        pairs[role] = _opposite_label_pairs(
            relabeled, seed=seed + ROLE_OFFSETS[role] + 97
        )
        assignments.extend(
            {
                "dataset": prepared.name,
                "role": role,
                "seed": seed,
                "example_id": example.example_id,
                "real_label": int(real_label),
                "random_label": int(random_label),
            }
            for example, real_label, random_label in zip(rows, original, assigned)
        )
    return RandomizedSelectivityData(
        examples=randomized,
        pairs=pairs,
        assignment_rows=tuple(assignments),
    )


__all__ = [
    "PreparedSelectivityData",
    "RandomizedSelectivityData",
    "prepare_ait_data",
    "prepare_toy_data",
    "randomize_data",
]
