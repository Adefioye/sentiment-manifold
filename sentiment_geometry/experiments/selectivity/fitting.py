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
    tuning_trials: tuple[dict[str, Any], ...] = ()


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
    trial_rows: list[dict[str, Any]] = []
    for trial_index, candidate in enumerate(search.candidates()):
        probe = fit_logistic_probe(train_x, train_y, config=candidate, seed=seed)
        evaluation = evaluate_binary_probe(
            training_midpoint_scores=probe.midpoint_scores(train_x),
            training_labels=train_y,
            native_scores=probe.native_scores(validation_x),
            midpoint_scores=probe.midpoint_scores(validation_x),
            labels=validation_y,
            native_threshold=probe.native_threshold,
        )
        hyperparameters = asdict(candidate)
        candidates.append(
            SelectedProbe(probe, hyperparameters, evaluation.native_balanced_accuracy)
        )
        trial_rows.append(
            {
                "trial_index": trial_index,
                "hyperparameters": hyperparameters,
                "validation_native_accuracy": evaluation.native_accuracy,
                "validation_native_balanced_accuracy": evaluation.native_balanced_accuracy,
                "validation_midpoint_accuracy": evaluation.midpoint_accuracy,
                "validation_midpoint_balanced_accuracy": (
                    evaluation.midpoint_balanced_accuracy
                ),
                "validation_loss": None,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row.validation_native_balanced_accuracy,
            -float(row.hyperparameters["c"]),
        ),
    )
    selected_index = candidates.index(selected)
    return replace(
        selected,
        tuning_trials=tuple(
            {**row, "selected": index == selected_index}
            for index, row in enumerate(trial_rows)
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
    show_progress: bool = False,
    progress_description: str | None = None,
    progress_leave: bool = False,
) -> SelectedProbe:
    candidates: list[SelectedProbe] = []
    trial_rows: list[dict[str, Any]] = []
    for trial_index, candidate in enumerate(search.candidates()):
        probe = fit_mlp1_probe(
            train_x,
            train_y,
            validation_x,
            validation_y,
            config=candidate,
            seed=seed,
            show_progress=show_progress,
            progress_description=(
                f"{progress_description or 'Tune MLP-1'} trial {trial_index + 1}"
            ),
            progress_leave=progress_leave,
        )
        evaluation = evaluate_binary_probe(
            training_midpoint_scores=probe.midpoint_scores(train_x),
            training_labels=train_y,
            native_scores=probe.native_scores(validation_x),
            midpoint_scores=probe.midpoint_scores(validation_x),
            labels=validation_y,
            native_threshold=probe.native_threshold,
        )
        hyperparameters = asdict(candidate)
        candidates.append(
            SelectedProbe(probe, hyperparameters, evaluation.native_balanced_accuracy)
        )
        trial_rows.append(
            {
                "trial_index": trial_index,
                "hyperparameters": hyperparameters,
                "validation_native_accuracy": evaluation.native_accuracy,
                "validation_native_balanced_accuracy": evaluation.native_balanced_accuracy,
                "validation_midpoint_accuracy": evaluation.midpoint_accuracy,
                "validation_midpoint_balanced_accuracy": (
                    evaluation.midpoint_balanced_accuracy
                ),
                "validation_loss": float(
                    probe.diagnostics["best_validation_loss"]
                ),
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            row.validation_native_balanced_accuracy,
            -int(row.hyperparameters["hidden_size"]),
            -float(row.hyperparameters["weight_decay"]),
        ),
    )
    selected_index = candidates.index(selected)
    return replace(
        selected,
        tuning_trials=tuple(
            {**row, "selected": index == selected_index}
            for index, row in enumerate(trial_rows)
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
    show_progress: bool = False,
    progress_description: str | None = None,
    progress_leave: bool = False,
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
            show_progress=show_progress,
            progress_description=progress_description,
            progress_leave=progress_leave,
        )
    raise ValueError(f"Unsupported non-causal probe method: {method}")


def fit_selected_real_probe(
    method: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    hyperparameters: Mapping[str, Any],
    seed: int,
    show_progress: bool = False,
    progress_description: str | None = None,
    progress_leave: bool = False,
) -> BinaryProbe:
    """Fit one real-task probe after validation tuning has been frozen."""

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
        return fit_mlp1_probe(
            train_x,
            train_y,
            validation_x,
            validation_y,
            config=MLP1ProbeConfig(**dict(hyperparameters)),
            seed=seed,
            show_progress=show_progress,
            progress_description=progress_description,
            progress_leave=progress_leave,
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
    show_progress: bool = False,
    progress_description: str | None = None,
    progress_leave: bool = False,
) -> FittedDAS:
    validation_evaluator = DirectionalPatchingEvaluator(
        adapter,
        list(validation_pairs),
        layer=layer,
        answers=answers,
        position="final",
        batch_size=evaluation_batch_size,
        show_progress=show_progress,
        progress_description=f"{progress_description or 'DAS'} validation",
        progress_leave=progress_leave,
    )

    def validate_epoch(direction: np.ndarray) -> dict[str, float]:
        result = validation_evaluator.evaluate(direction, show_progress=False)
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
            show_progress=show_progress,
            progress_description=progress_description,
            progress_leave=progress_leave,
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
            show_progress=show_progress,
            progress_description=progress_description,
            progress_leave=progress_leave,
        )
    return FittedDAS(
        fitted,
        validation_evaluator.evaluate(
            fitted.direction,
            show_progress=show_progress,
            progress_description=f"{progress_description or 'DAS'} validation",
            progress_leave=progress_leave,
        ),
    )


__all__ = [
    "FittedDAS",
    "SelectedProbe",
    "fit_final_token_das",
    "fit_selected_control_probe",
    "fit_selected_real_probe",
    "select_logistic_probe",
    "select_mlp1_probe",
]
