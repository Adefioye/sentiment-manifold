"""Direction fitting and checkpoint reuse for token-position comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...datasets.types import CounterfactualPair, TextExample
from ...fitting_methods import DirectionArtifact, create_fitter
from ...fitting_methods.base import FitResult
from ...fitting_methods.das import DASFitter, DASTrainingConfig
from ...models import CausalLMAdapter
from ...models.config import ModelConfig
from ...persistence import checkpoint_variant_dir
from .config import SentimentPositionExperimentConfig

ARTIFACT_SCHEMA_VERSION = 1
AnswerSpec = dict[int, tuple[str, ...]]


@dataclass(frozen=True)
class DirectionFitRequest:
    examples: Sequence[TextExample]
    pairs: Sequence[CounterfactualPair]
    activations: np.ndarray
    labels: np.ndarray
    method: str
    fit_position: str
    layer: int
    answers: AnswerSpec


@dataclass(frozen=True)
class FittedDirection:
    artifact: DirectionArtifact
    checkpoint_path: Path


class DirectionFitService:
    """Fit or restore one normalized sentiment direction."""

    def __init__(
        self,
        *,
        config: SentimentPositionExperimentConfig,
        adapter: CausalLMAdapter,
        model: ModelConfig,
        runtime: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.model = model
        self.runtime = runtime

    def fit(self, request: DirectionFitRequest) -> FittedDirection:
        hyperparameters = self._hyperparameters(request.method)
        path = self._checkpoint_path(request, hyperparameters)
        artifact = self._load_compatible(path, request, hyperparameters)
        if artifact is None:
            artifact = self._train(request, hyperparameters)
            artifact.save(path)
        return FittedDirection(artifact=artifact, checkpoint_path=path)

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
            }
        return {}

    def _checkpoint_path(
        self,
        request: DirectionFitRequest,
        hyperparameters: Mapping[str, Any],
    ) -> Path:
        fingerprint = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model": self.model.name,
            "model_revision": self.runtime.get("resolved_model_revision"),
            "tokenizer_revision": self.runtime.get("resolved_tokenizer_revision"),
            "prepend_bos": self.model.prepend_bos,
            "fit_position": request.fit_position,
            "examples": [
                {"example_id": row.example_id, "text": row.text, "label": row.label}
                for row in request.examples
            ],
            "hyperparameters": dict(hyperparameters),
        }
        directory = checkpoint_variant_dir(
            self.config.sweep.checkpoint_dir,
            model_name=self.model.name,
            phase="sentiment-position-comparison",
            method=f"{request.fit_position}-{request.method}",
            fingerprint_payload=fingerprint,
        )
        return directory / f"layer{request.layer:02d}.npz"

    def _load_compatible(
        self,
        path: Path,
        request: DirectionFitRequest,
        hyperparameters: Mapping[str, Any],
    ) -> DirectionArtifact | None:
        if not self.config.sweep.resume or not path.exists():
            return None
        candidate = DirectionArtifact.load(path)
        compatible = (
            candidate.layer == request.layer
            and candidate.method == request.method
            and candidate.model_name == self.model.hub_name
            and candidate.metadata.get("artifact_schema_version") == ARTIFACT_SCHEMA_VERSION
            and candidate.metadata.get("fit_position") == request.fit_position
            and candidate.metadata.get("fit_hyperparameters") == dict(hyperparameters)
        )
        return candidate if compatible else None

    def _train(
        self,
        request: DirectionFitRequest,
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
                "fit_position": request.fit_position,
                "fit_hyperparameters": dict(hyperparameters),
                "model_revision": self.runtime.get("resolved_model_revision"),
                "tokenizer_revision": self.runtime.get("resolved_tokenizer_revision"),
            },
        )

    def _fit_das(self, request: DirectionFitRequest) -> FitResult:
        if self.config.das.implementation != "tigges_rotation":
            raise ValueError("DAS requires das.implementation=tigges_rotation")
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
        extraction_position = "focus" if request.fit_position == "adjective" else "final"
        return fitter.fit(
            self.adapter,
            list(request.pairs),
            layer=request.layer,
            answers=request.answers,
            position=extraction_position,
        )


__all__ = ["DirectionFitRequest", "DirectionFitService", "FittedDirection"]
