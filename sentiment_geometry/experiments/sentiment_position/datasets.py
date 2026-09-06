"""Prepare fitting and causal-evaluation data for position comparisons."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ...datasets import build_toy_evaluation_sets, load_hf_directed_pairs, load_toy_movie_review
from ...datasets.toy_movie_review import ToyEvaluationSet, ToyMovieReview
from ...datasets.types import CounterfactualPair, TextExample
from ...models import CausalLMAdapter
from ...models.config import ModelConfig
from .config import REQUIRED_TOY_EVALUATIONS, SentimentPositionExperimentConfig

SST_ANSWERS = {1: (" Positive",), 0: (" Negative",)}
AnswerSpec = dict[int, tuple[str, ...]]


@dataclass(frozen=True)
class CausalEvaluation:
    name: str
    pairs: tuple[CounterfactualPair, ...]
    answers: AnswerSpec
    examples: tuple[TextExample, ...]


@dataclass(frozen=True)
class PreparedSentimentData:
    raw_toy: ToyMovieReview
    filtered_toy: ToyMovieReview
    train_examples: tuple[TextExample, ...]
    train_pairs: tuple[CounterfactualPair, ...]
    toy_evaluations: Mapping[str, ToyEvaluationSet]
    evaluations: Mapping[str, CausalEvaluation]
    sst_config: str
    sst_resolved_revision: str | None

    @property
    def answers_by_dataset(self) -> dict[str, AnswerSpec]:
        return {name: evaluation.answers for name, evaluation in self.evaluations.items()}


class SentimentDatasetLoader:
    """Load and validate all data required by one model's experiment."""

    def __init__(
        self,
        *,
        config: SentimentPositionExperimentConfig,
        model: ModelConfig,
        adapter: CausalLMAdapter,
    ) -> None:
        self.config = config
        self.model = model
        self.adapter = adapter

    def load(self) -> PreparedSentimentData:
        raw_toy = load_toy_movie_review(self.config.data.toy_config)
        filtered_toy = raw_toy.tokenizer_filtered(self.adapter.tokenizer)
        train_examples = tuple(
            row for row in filtered_toy.train if self.adapter.focus_is_single_token(row)
        )
        allowed_train = {row.example_id for row in train_examples}
        train_pairs = tuple(
            pair
            for pair in filtered_toy.paired("train")
            if pair.clean.example_id in allowed_train and pair.corrupted.example_id in allowed_train
        )
        if not train_examples or not train_pairs:
            raise RuntimeError(f"No tokenizer-compatible Toy training data for {self.model.name}")

        toy_evaluations = build_toy_evaluation_sets(
            raw_toy,
            self.adapter.tokenizer,
            prepend_bos=self.model.prepend_bos,
        )
        self._validate_required_toy_evaluations(toy_evaluations)
        sst_config = self.config.data.sst_configs[self.model.name]
        sst_pairs = tuple(
            load_hf_directed_pairs(
                self.config.data.sst_repo_id,
                config_name=sst_config,
                split=self.config.data.sst_split,
                revision=self.config.data.sst_revision,
                token=_hugging_face_token(self.config.data.hf_token_env),
                max_pairs=self.config.data.sst_max_directed_cases,
            )
        )
        sst_examples = tuple(
            {
                example.example_id: example
                for pair in sst_pairs
                for example in (pair.clean, pair.corrupted)
            }.values()
        )
        evaluations = {
            name: CausalEvaluation(
                name=name,
                pairs=evaluation.pairs,
                answers=evaluation.answers,
                examples=evaluation.examples,
            )
            for name, evaluation in toy_evaluations.items()
        }
        evaluations["sst"] = CausalEvaluation(
            name="sst",
            pairs=sst_pairs,
            answers=SST_ANSWERS,
            examples=sst_examples,
        )
        resolved_revision = sst_pairs[0].clean.metadata.get("resolved_dataset_revision")
        return PreparedSentimentData(
            raw_toy=raw_toy,
            filtered_toy=filtered_toy,
            train_examples=train_examples,
            train_pairs=train_pairs,
            toy_evaluations=toy_evaluations,
            evaluations=evaluations,
            sst_config=sst_config,
            sst_resolved_revision=(
                str(resolved_revision) if resolved_revision is not None else None
            ),
        )

    @staticmethod
    def _validate_required_toy_evaluations(
        evaluations: Mapping[str, ToyEvaluationSet],
    ) -> None:
        missing = [name for name in REQUIRED_TOY_EVALUATIONS if name not in evaluations]
        empty = [
            name
            for name in REQUIRED_TOY_EVALUATIONS
            if name in evaluations and not evaluations[name].pairs
        ]
        if missing or empty:
            raise RuntimeError(
                f"Required Toy evaluation data is incomplete; missing={missing}, empty={empty}"
            )


def _hugging_face_token(variable_name: str) -> str | None:
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


__all__ = [
    "SST_ANSWERS",
    "CausalEvaluation",
    "PreparedSentimentData",
    "SentimentDatasetLoader",
]
