"""Probe selection and final-token DAS fitting for selectivity experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

import numpy as np

from ...datasets import CounterfactualPair
from ...evaluation import DirectionalPatchingEvaluator, PatchingResult
from ...fitting_methods.base import FitResult
from ...fitting_methods.das import DASFitter, DASTrainingConfig
from ...models import CausalLMAdapter
from ...probes import (
    BinaryProbe,
    LogisticProbeConfig,
    MLP1ProbeConfig,
    evaluate_binary_probe,
    fit_logistic_probe,
    fit_mean_difference_probe,
    fit_mlp1_probe,
)
from .config import DASSelectivityConfig, LogisticSearchConfig, MLP1SearchConfig


@dataclass(frozen=True)
class SelectedProbe:
    probe: BinaryProbe
    hyperparameters: dict[str, Any]
    validation_native_balanced_accuracy: float


def select_logistic_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    search: LogisticSearchConfig,
    seed: int,
) -> SelectedProbe:
    candidates: list[SelectedProbe] = []
    for candidate in search.candidates():
        probe = fit_logistic_probe(train_x, train_y, config=candidate, seed=seed)
        evaluation = evaluate_binary_probe(
            training_midpoint_scores=probe.midpoint_scores(train_x),
            training_labels=train_y,
            native_scores=probe.native_scores(validation_x),
            midpoint_scores=probe.midpoint_scores(validation_x),
            labels=validation_y,
            native_threshold=probe.native_threshold,
        )
        candidates.append(
            SelectedProbe(probe, asdict(candidate), evaluation.native_balanced_accuracy)
        )
    return max(
        candidates,
        key=lambda row: (
            row.validation_native_balanced_accuracy,
            -float(row.hyperparameters["c"]),
        ),
    )


def select_mlp1_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    search: MLP1SearchConfig,
    seed: int,
) -> SelectedProbe:
    candidates: list[SelectedProbe] = []
    for candidate in search.candidates():
        probe = fit_mlp1_probe(
            train_x,
            train_y,
            validation_x,
            validation_y,
            config=candidate,
            seed=seed,
        )
        evaluation = evaluate_binary_probe(
            training_midpoint_scores=probe.midpoint_scores(train_x),
            training_labels=train_y,
            native_scores=probe.native_scores(validation_x),
            midpoint_scores=probe.midpoint_scores(validation_x),
            labels=validation_y,
            native_threshold=probe.native_threshold,
        )
        candidates.append(
            SelectedProbe(probe, asdict(candidate), evaluation.native_balanced_accuracy)
        )
    return max(
        candidates,
        key=lambda row: (
            row.validation_native_balanced_accuracy,
            -int(row.hyperparameters["hidden_size"]),
            -float(row.hyperparameters["weight_decay"]),
        ),
    )


def fit_selected_control_probe(
    method: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    hyperparameters: Mapping[str, Any],
    seed: int,
    training_epochs: int | None = None,
) -> BinaryProbe:
    if method == "mean_diff":
        return fit_mean_difference_probe(train_x, train_y)
    if method == "logistic_regression":
        return fit_logistic_probe(
            train_x,
            train_y,
            config=LogisticProbeConfig(**dict(hyperparameters)),
            seed=seed,
        )
    if method == "mlp1":
        config = MLP1ProbeConfig(**dict(hyperparameters))
        if training_epochs is None or training_epochs < 1:
            raise ValueError("MLP-1 control requires the paired real training duration")
        config = replace(
            config,
            max_epochs=training_epochs,
            patience=training_epochs + 1,
        )
        return fit_mlp1_probe(
            train_x,
            train_y,
            validation_x,
            validation_y,
            config=config,
            seed=seed,
            select_final_checkpoint=True,
        )
    raise ValueError(f"Unsupported non-causal probe method: {method}")


@dataclass(frozen=True)
class FittedDAS:
    fit_result: FitResult
    validation: PatchingResult


def fit_final_token_das(
    adapter: CausalLMAdapter,
    train_pairs: tuple[CounterfactualPair, ...],
    validation_pairs: tuple[CounterfactualPair, ...],
    *,
    layer: int,
    answers: dict[int, tuple[str, ...]],
    config: DASSelectivityConfig,
    seed: int,
    evaluation_batch_size: int,
    fixed_epochs: int | None = None,
) -> FittedDAS:
    validation_evaluator = DirectionalPatchingEvaluator(
        adapter,
        list(validation_pairs),
        layer=layer,
        answers=answers,
        position="final",
        batch_size=evaluation_batch_size,
    )

    def validate_epoch(direction: np.ndarray) -> dict[str, float]:
        result = validation_evaluator.evaluate(direction)
        return {
            "validation_loss": 1.0 - result.iia,
            "validation_iia": result.iia,
            "validation_recovery": result.recovery,
            "validation_logit_flip": result.flip_rate,
        }

    fitter = DASFitter(
        DASTrainingConfig(
            epochs=fixed_epochs or config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            batch_size=config.batch_size,
            max_grad_norm=config.max_grad_norm,
            seed=seed,
            objective=config.objective,
        )
    )
    if fixed_epochs is None:
        fitted = fitter.fit(
            adapter,
            list(train_pairs),
            layer=layer,
            answers=answers,
            position="final",
            epoch_validator=validate_epoch,
            checkpoint_metric=config.checkpoint_metric,
        )
    else:
        fitted = fitter.fit(
            adapter,
            list(train_pairs),
            layer=layer,
            answers=answers,
            position="final",
            checkpoint_metric="post_epoch_train_loss",
            select_final_epoch=True,
        )
    return FittedDAS(fitted, validation_evaluator.evaluate(fitted.direction))


__all__ = [
    "FittedDAS",
    "SelectedProbe",
    "fit_final_token_das",
    "fit_selected_control_probe",
    "select_logistic_probe",
    "select_mlp1_probe",
]
