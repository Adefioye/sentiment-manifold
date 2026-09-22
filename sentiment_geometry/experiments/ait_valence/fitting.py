"""Fit and checkpoint AIT valence directions with sentiment-compatible artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...datasets import CounterfactualPair, TextExample
from ...evaluation import DirectionalPatchingEvaluator
from ...fitting_methods import DirectionArtifact, create_fitter
from ...fitting_methods.das import DASFitter, DASTrainingConfig
from ...models import CausalLMAdapter, ModelConfig
from ...persistence import checkpoint_variant_dir
from .config import AITValenceExperimentConfig

ARTIFACT_SCHEMA_VERSION = 2
AnswerSpec = dict[int, tuple[str, ...]]


@dataclass(frozen=True)
class AITDirectionFitRequest:
    examples: Sequence[TextExample]
    train_pairs: Sequence[CounterfactualPair]
    eval_pairs: Sequence[CounterfactualPair]
    activations: np.ndarray
    labels: np.ndarray
    method: str
    layer: int
    answers: AnswerSpec

    @property
    def representation(self) -> str:
        return "all_tokens" if self.method == "das" else "masked_mean"


@dataclass(frozen=True)
class FittedAITDirection:
    artifact: DirectionArtifact
    checkpoint_path: Path


class AITDirectionFitService:
    """Fit or restore one positive-oriented AIT valence direction."""

    def __init__(
        self,
        *,
        config: AITValenceExperimentConfig,
        adapter: CausalLMAdapter,
        model: ModelConfig,
        runtime: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.model = model
        self.runtime = runtime
        self._validation_layer: int | None = None
        self._validation_evaluator: DirectionalPatchingEvaluator | None = None

    def fit(self, request: AITDirectionFitRequest) -> FittedAITDirection:
        hyperparameters = self._hyperparameters(request.method)
        path = self._checkpoint_path(request, hyperparameters)
        artifact = self._load_compatible(path, request, hyperparameters)
        if artifact is None:
            artifact = self._train(request, hyperparameters)
            artifact.save(path)
        return FittedAITDirection(artifact=artifact, checkpoint_path=path)

    def _hyperparameters(self, method: str) -> dict[str, Any]:
        if method == "logistic_regression":
            return {
                "c": self.config.fitting.logistic_c,
                "solver": self.config.fitting.logistic_solver,
                "max_iter": self.config.fitting.logistic_max_iter,
                "tol": self.config.fitting.logistic_tol,
                "random_state": self.config.seed,
            }
        if method == "das":
            return {
                "dimension": 1,
                "epochs": self.config.das.epochs,
                "learning_rate": self.config.das.learning_rate,
                "weight_decay": self.config.das.weight_decay,
                "batch_size": self.config.das.batch_size,
                "max_grad_norm": self.config.das.max_grad_norm,
                "seed": self.config.seed,
                "implementation": self.config.das.implementation,
                "checkpoint_role": self.config.selection.das_checkpoint_split,
                "checkpoint_metric": self.config.selection.das_checkpoint_metric,
            }
        return {}

    def _checkpoint_path(
        self,
        request: AITDirectionFitRequest,
        hyperparameters: Mapping[str, Any],
    ) -> Path:
        fingerprint = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model": self.model.name,
            "model_revision": self.runtime.get("resolved_model_revision"),
            "tokenizer_revision": self.runtime.get("resolved_tokenizer_revision"),
            "dataset_repo_id": self.config.data.repo_id,
            "dataset_config": self.config.data.matched_config,
            "requested_dataset_revision": self.runtime.get(
                "requested_dataset_revision", self.config.data.revision
            ),
            "resolved_dataset_revision": self.runtime.get("resolved_dataset_revision"),
            "prepend_bos": self.model.prepend_bos,
            "method": request.method,
            "representation": request.representation,
            "layer": request.layer,
            "training_examples": [
                {
                    "example_id": example.example_id,
                    "text": example.text,
                    "label": example.label,
                }
                for example in request.examples
            ],
            "training_pairs": [
                {
                    "case_id": pair.clean.metadata.get("case_id"),
                    "clean_id": pair.clean.example_id,
                    "corrupted_id": pair.corrupted.example_id,
                }
                for pair in request.train_pairs
            ],
            "checkpoint_pairs": [
                {
                    "case_id": pair.clean.metadata.get("case_id"),
                    "clean_id": pair.clean.example_id,
                    "corrupted_id": pair.corrupted.example_id,
                }
                for pair in request.eval_pairs
            ]
            if request.method == "das"
            else None,
            "answers": {str(label): list(values) for label, values in request.answers.items()},
            "hyperparameters": dict(hyperparameters),
        }
        directory = checkpoint_variant_dir(
            self.config.sweep.checkpoint_dir,
            model_name=self.model.name,
            phase="ait-valence-directions",
            method=request.method,
            fingerprint_payload=fingerprint,
        )
        return directory / f"layer{request.layer:02d}.npz"

    def _load_compatible(
        self,
        path: Path,
        request: AITDirectionFitRequest,
        hyperparameters: Mapping[str, Any],
    ) -> DirectionArtifact | None:
        if not self.config.sweep.resume or not path.is_file():
            return None
        candidate = DirectionArtifact.load(path)
        compatible = (
            candidate.layer == request.layer
            and candidate.method == request.method
            and candidate.model_name == self.model.hub_name
            and candidate.metadata.get("artifact_schema_version") == ARTIFACT_SCHEMA_VERSION
            and candidate.metadata.get("representation") == request.representation
            and candidate.metadata.get("fit_hyperparameters") == dict(hyperparameters)
        )
        return candidate if compatible else None

    def _train(
        self,
        request: AITDirectionFitRequest,
        hyperparameters: Mapping[str, Any],
    ) -> DirectionArtifact:
        if request.method == "das":
            result = self._fit_das(request)
        elif request.method == "logistic_regression":
            result = create_fitter(request.method, **dict(hyperparameters)).fit(
                request.activations, request.labels
            )
        else:
            result = create_fitter(request.method).fit(request.activations, request.labels)
        return DirectionArtifact(
            method=request.method,
            model_name=self.model.hub_name,
            layer=request.layer,
            vector=result.direction,
            metadata={
                **result.diagnostics,
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "domain": "ait_valence",
                "representation": request.representation,
                "fit_position": request.representation,
                "fit_hyperparameters": dict(hyperparameters),
                "orientation_convention": "negative_to_positive",
                "orientation_reference": "ait_train_class_mean_difference",
                "n_training_examples": len(request.examples),
                "n_training_directed_cases": len(request.train_pairs),
                "dataset_repo_id": self.config.data.repo_id,
                "dataset_config": self.config.data.matched_config,
                "requested_dataset_revision": self.runtime.get(
                    "requested_dataset_revision", self.config.data.revision
                ),
                "resolved_dataset_revision": self.runtime.get(
                    "resolved_dataset_revision"
                ),
                "model_revision": self.runtime.get("resolved_model_revision"),
                "tokenizer_revision": self.runtime.get("resolved_tokenizer_revision"),
            },
        )

    def _fit_das(self, request: AITDirectionFitRequest):
        fitter = DASFitter(
            DASTrainingConfig(
                epochs=self.config.das.epochs,
                learning_rate=self.config.das.learning_rate,
                weight_decay=self.config.das.weight_decay,
                batch_size=self.config.das.batch_size,
                max_grad_norm=self.config.das.max_grad_norm,
                seed=self.config.seed,
            )
        )
        if self._validation_layer != request.layer:
            self._validation_evaluator = DirectionalPatchingEvaluator(
                self.adapter,
                list(request.eval_pairs),
                layer=request.layer,
                answers=request.answers,
                position="all",
                batch_size=self.model.batch_size,
            )
            self._validation_layer = request.layer
        assert self._validation_evaluator is not None
        validation_evaluator = self._validation_evaluator

        def validate_epoch(direction: np.ndarray) -> dict[str, float]:
            validation = validation_evaluator.evaluate(direction)
            return {
                "validation_loss": 1.0 - validation.recovery,
                "validation_logit_difference_percent": validation.recovery_percent,
                "validation_logit_flip_percent": validation.flip_percent,
                "validation_sign_flip_percent": validation.sign_flip_percent,
                "validation_corrupted_accuracy": validation.corrupted_accuracy,
                "validation_clean_accuracy": validation.clean_accuracy,
                "validation_patched_accuracy": validation.patched_accuracy,
            }

        return fitter.fit(
            self.adapter,
            list(request.train_pairs),
            layer=request.layer,
            answers=request.answers,
            position="all",
            epoch_validator=validate_epoch,
            checkpoint_metric="validation_loss",
        )


__all__ = ["AITDirectionFitRequest", "AITDirectionFitService", "FittedAITDirection"]
