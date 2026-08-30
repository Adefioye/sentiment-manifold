"""Leakage-safe validation tuning for sentiment direction methods."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .artifacts import DirectionArtifact
from .config import ReproductionConfig
from .data import load_toy_movie_review, pair_toy_examples
from .devices import clear_device_cache, resolve_device
from .directions import create_fitter
from .directions.das import DASFitter, DASTrainingConfig
from .evaluation import evaluate_directional_patching, projection_accuracy
from .evaluation.projections import projection_threshold
from .experiment import ARTIFACT_SCHEMA_VERSION, run_reproduction
from .models import CausalLMAdapter
from .types import TextExample


TUNABLE_METHODS = (
    "mean_diff",
    "kmeans",
    "logistic_regression",
    "pca",
    "das",
    "das2d",
    "das3d",
)
TUNING_METRICS = (
    "toy_validation_logit_diff_percent",
    "toy_validation_logit_flip_percent",
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")


def _write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def split_toy_training_examples(
    examples: list[TextExample], *, validation_fraction: float, seed: int
) -> tuple[list[TextExample], list[TextExample]]:
    """Make a deterministic stratified split without touching Toy test examples."""

    if not 0 < validation_fraction < 1:
        raise ValueError("tuning.validation_fraction must be strictly between 0 and 1")
    generator = np.random.default_rng(seed)
    validation_ids: set[str] = set()
    for label in (0, 1):
        labelled = [example for example in examples if example.label == label]
        if len(labelled) < 2:
            raise ValueError("Each Toy training class needs at least two examples for tuning")
        indices = generator.permutation(len(labelled))
        n_validation = min(
            len(labelled) - 1,
            max(1, int(round(validation_fraction * len(labelled)))),
        )
        validation_ids.update(labelled[index].example_id for index in indices[:n_validation])
    fit = [example for example in examples if example.example_id not in validation_ids]
    validation = [example for example in examples if example.example_id in validation_ids]
    return fit, validation


def tuning_grid(config: ReproductionConfig, method: str) -> list[dict[str, Any]]:
    """Expand the configured grid; random seeds are trials, not selectable hyperparameters."""

    if method not in TUNABLE_METHODS:
        raise ValueError(f"Unsupported tuning method {method!r}; choose from {TUNABLE_METHODS}")
    seeds = [int(seed) for seed in config.tuning.seeds]
    if not seeds:
        raise ValueError("tuning.seeds must contain at least one seed")
    if method in {"mean_diff", "pca"}:
        return [{"seed": config.seed}]
    if method == "kmeans":
        return [
            {"seed": seed, "n_init": int(n_init)}
            for n_init, seed in itertools.product(config.tuning.kmeans_n_init, seeds)
        ]
    if method == "logistic_regression":
        return [
            {
                "seed": seed,
                "c": float(c),
                "solver": config.fitting.logistic_solver,
                "max_iter": config.fitting.logistic_max_iter,
                "tol": config.fitting.logistic_tol,
            }
            for c, seed in itertools.product(config.tuning.logistic_c, seeds)
        ]
    return [
        {
            "seed": int(seed),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "max_grad_norm": float(max_grad_norm),
        }
        for learning_rate, weight_decay, epochs, batch_size, max_grad_norm, seed in itertools.product(
            config.tuning.das_learning_rate,
            config.tuning.das_weight_decay,
            config.tuning.das_epochs,
            config.tuning.das_batch_size,
            config.tuning.das_max_grad_norm,
            seeds,
        )
    ]


def _hyperparameters(parameters: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in parameters.items() if key != "seed"}


def _trial_id(method: str, layer: int, parameters: dict[str, Any]) -> str:
    payload = json.dumps(
        {"method": method, "layer": layer, **parameters}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def select_validation_configuration(trials: pd.DataFrame, *, selection_metric: str) -> pd.DataFrame:
    """Select a layer/hyperparameter configuration by mean validation score across seeds."""

    if selection_metric not in TUNING_METRICS:
        raise ValueError(f"selection_metric must be one of {TUNING_METRICS}")
    required = {"model", "method", "layer", "hyperparameters", "seed", selection_metric}
    missing = sorted(required - set(trials.columns))
    if missing:
        raise ValueError(f"Cannot select a tuning configuration; missing columns: {missing}")
    valid = trials.dropna(subset=[selection_metric]).copy()
    if valid.empty:
        raise ValueError(f"No non-missing validation values for {selection_metric}")
    keys = ["model", "method", "layer", "hyperparameters"]
    scores = valid.groupby(keys, as_index=False)[selection_metric].agg(
        validation_mean="mean",
        validation_std=lambda values: values.std(ddof=0),
    )
    counts = (
        valid.groupby(keys, as_index=False)["seed"].nunique().rename(columns={"seed": "n_seeds"})
    )
    summary = scores.merge(counts, on=keys)
    selected_rows = []
    for _, method_rows in summary.groupby(["model", "method"], sort=False):
        selected_rows.append(
            method_rows.sort_values(
                ["validation_mean", "validation_std", "layer", "hyperparameters"],
                ascending=[False, True, True, True],
                kind="stable",
            ).iloc[0]
        )
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected.insert(4, "selection_metric", selection_metric)
    return selected


def _fit_trial(
    adapter: CausalLMAdapter,
    method: str,
    layer: int,
    parameters: dict[str, Any],
    fit_pairs,
    fit_activations: np.ndarray,
    fit_labels: np.ndarray,
    answers,
):
    seed = int(parameters["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if method.startswith("das"):
        dimension = {"das": 1, "das2d": 2, "das3d": 3}[method]
        fitter = DASFitter(
            DASTrainingConfig(
                epochs=int(parameters["epochs"]),
                learning_rate=float(parameters["learning_rate"]),
                weight_decay=float(parameters["weight_decay"]),
                batch_size=int(parameters["batch_size"]),
                max_grad_norm=float(parameters["max_grad_norm"]),
                seed=seed,
            ),
            dimension=dimension,
            name=method,
        )
        return fitter.fit(adapter, fit_pairs, layer=layer, answers=answers)
    if method == "kmeans":
        fitter = create_fitter(method, random_state=seed, n_init=int(parameters["n_init"]))
    elif method == "logistic_regression":
        fitter = create_fitter(
            method,
            random_state=seed,
            c=float(parameters["c"]),
            solver=str(parameters["solver"]),
            max_iter=int(parameters["max_iter"]),
            tol=float(parameters["tol"]),
        )
    else:
        fitter = create_fitter(method)
    return fitter.fit(fit_activations, fit_labels)


def run_tuning(config: ReproductionConfig, method: str) -> Path:
    """Tune one method using only a held-out subset of ToyMovieReview training data."""

    if config.tuning.selection_metric not in TUNING_METRICS:
        raise ValueError(f"tuning.selection_metric must be one of {TUNING_METRICS}")
    device_spec = resolve_device(config.model.device, config.model.dtype)
    adapter = CausalLMAdapter.from_pretrained(
        config.model.hub_name,
        device_spec,
        revision=config.model.revision,
        prepend_bos=config.model.prepend_bos,
    )
    run_dir = Path(config.experiment.output_dir) / _slug(config.model.name) / method
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = config.to_dict()
    resolved["runtime"] = adapter.provenance()
    resolved["tuned_method"] = method
    (run_dir / "tuning_resolved_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True)
    )

    toy = load_toy_movie_review(config.data.toy_config).tokenizer_filtered(adapter.tokenizer)
    retained = [example for example in toy.train if adapter.focus_is_single_token(example)]
    fit_examples, validation_examples = split_toy_training_examples(
        retained,
        validation_fraction=config.tuning.validation_fraction,
        seed=config.seed,
    )
    fit_pairs = pair_toy_examples(fit_examples)
    validation_pairs = pair_toy_examples(validation_examples)
    if not fit_pairs or not validation_pairs:
        raise RuntimeError("Tuning split did not produce fit and validation counterfactual pairs")
    fit_ids = {example.example_id for example in fit_examples}
    split_rows = [
        {
            "example_id": example.example_id,
            "label": example.label,
            "adjective": example.metadata.get("adjective"),
            "split": "fit" if example.example_id in fit_ids else "validation",
        }
        for example in retained
    ]
    _write_rows(split_rows, run_dir / "tuning_split.csv")

    fit_labels = np.asarray([example.label for example in fit_examples])
    validation_labels = np.asarray([example.label for example in validation_examples])
    trial_rows: list[dict] = []
    patching_rows: list[dict] = []
    loss_rows: list[dict] = []
    candidates = tuning_grid(config, method)
    layers = config.layers_for(adapter.n_layers)
    if not candidates:
        raise ValueError(f"The tuning grid for {method!r} is empty")
    if not layers:
        raise ValueError("experiment.layers must contain at least one layer for tuning")
    for layer in tqdm(layers, desc=f"tune {method} layer boundaries"):
        fit_activations = adapter.extract_activations(
            fit_examples, layer, position="focus", batch_size=config.model.batch_size
        )
        validation_activations = adapter.extract_activations(
            validation_examples, layer, position="focus", batch_size=config.model.batch_size
        )
        for parameters in candidates:
            result = _fit_trial(
                adapter,
                method,
                layer,
                parameters,
                fit_pairs,
                fit_activations,
                fit_labels,
                toy.answers,
            )
            trial_id = _trial_id(method, layer, parameters)
            hyperparameters = _hyperparameters(parameters)
            artifact = DirectionArtifact(
                method=method,
                model_name=config.model.hub_name,
                layer=layer,
                vector=result.direction,
                metadata={
                    **result.diagnostics,
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "phase": "validation_tuning",
                    "trial_id": trial_id,
                    "fit_example_ids": [example.example_id for example in fit_examples],
                    "validation_example_ids": [
                        example.example_id for example in validation_examples
                    ],
                    "hyperparameters": hyperparameters,
                    "seed": int(parameters["seed"]),
                },
            )
            artifact_path = run_dir / "trial_directions" / f"{trial_id}.npz"
            artifact.save(artifact_path)
            validation = evaluate_directional_patching(
                adapter,
                validation_pairs,
                artifact.vector,
                layer=layer,
                answers=toy.answers,
                position="all",
                batch_size=config.model.batch_size,
            )
            if artifact.vector.ndim == 1:
                threshold = projection_threshold(fit_activations, fit_labels, artifact.vector)
                validation_projection = projection_accuracy(
                    validation_activations,
                    validation_labels,
                    artifact.vector,
                    threshold,
                )
            else:
                validation_projection = float("nan")
            hyperparameters_json = json.dumps(
                hyperparameters, sort_keys=True, separators=(",", ":")
            )
            trial_rows.append(
                {
                    "model": config.model.name,
                    "method": method,
                    "layer": layer,
                    "trial_id": trial_id,
                    "seed": int(parameters["seed"]),
                    "hyperparameters": hyperparameters_json,
                    "toy_validation_projection_accuracy": validation_projection,
                    "toy_validation_logit_diff_percent": validation.recovery_percent,
                    "toy_validation_logit_flip_percent": validation.flip_percent,
                    "toy_validation_sign_flip_percent": validation.sign_flip_percent,
                    "toy_validation_corrupted_accuracy": validation.corrupted_accuracy,
                    "toy_validation_clean_accuracy": validation.clean_accuracy,
                    "toy_validation_patched_accuracy": validation.patched_accuracy,
                    "n_fit_examples": len(fit_examples),
                    "n_validation_examples": len(validation_examples),
                    "n_fit_pairs": len(fit_pairs),
                    "n_validation_pairs": len(validation_pairs),
                    "artifact_path": str(artifact_path),
                }
            )
            patching_rows.extend(
                {
                    "model": config.model.name,
                    "method": method,
                    "layer": layer,
                    "trial_id": trial_id,
                    "seed": int(parameters["seed"]),
                    "hyperparameters": hyperparameters_json,
                    **record,
                }
                for record in validation.records
            )
            for loss in result.diagnostics.get("loss_history", []):
                loss_rows.append(
                    {
                        "model": config.model.name,
                        "method": method,
                        "layer": layer,
                        "trial_id": trial_id,
                        "seed": int(parameters["seed"]),
                        "hyperparameters": hyperparameters_json,
                        **loss,
                    }
                )
            _write_rows(trial_rows, run_dir / "tuning_trials.csv")
            _write_rows(patching_rows, run_dir / "tuning_patching_records.csv")
            if loss_rows:
                _write_rows(loss_rows, run_dir / "tuning_das_losses.csv")
            clear_device_cache(device_spec.device)

    trials = pd.DataFrame(trial_rows)
    selected = select_validation_configuration(
        trials, selection_metric=config.tuning.selection_metric
    )
    selected.to_csv(run_dir / "selected_configs.csv", index=False)
    return run_dir


def apply_selected_configuration(
    config: ReproductionConfig,
    selected: pd.DataFrame,
    *,
    method: str | None = None,
    confirmation_seed: int | None = None,
) -> ReproductionConfig:
    """Freeze one selected validation configuration into a reproduction config."""

    required = {"model", "method", "layer", "hyperparameters"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"Selected configuration is missing columns: {missing}")
    if method is not None:
        selected = selected[selected["method"] == method]
    if len(selected) != 1:
        raise ValueError("Confirmation requires exactly one selected configuration row")
    row = selected.iloc[0]
    if str(row["model"]) != config.model.name:
        raise ValueError(
            f"Selection was tuned for model {row['model']!r}, but confirmation uses "
            f"{config.model.name!r}"
        )
    chosen_method = str(row["method"])
    hyperparameters = json.loads(str(row["hyperparameters"]))
    if not isinstance(hyperparameters, dict):
        raise ValueError("Selected hyperparameters must decode to a JSON object")
    config.experiment.methods = [chosen_method]
    config.experiment.layers = [int(row["layer"])]
    if confirmation_seed is not None:
        config.seed = int(confirmation_seed)
    if chosen_method == "kmeans":
        config.fitting.kmeans_n_init = int(hyperparameters["n_init"])
    elif chosen_method == "logistic_regression":
        config.fitting.logistic_c = float(hyperparameters["c"])
        config.fitting.logistic_solver = str(hyperparameters["solver"])
        config.fitting.logistic_max_iter = int(hyperparameters["max_iter"])
        config.fitting.logistic_tol = float(hyperparameters["tol"])
    elif chosen_method.startswith("das"):
        config.das.learning_rate = float(hyperparameters["learning_rate"])
        config.das.weight_decay = float(hyperparameters["weight_decay"])
        config.das.epochs = int(hyperparameters["epochs"])
        config.das.batch_size = int(hyperparameters["batch_size"])
        config.das.max_grad_norm = float(hyperparameters["max_grad_norm"])
    return config


def run_confirmation(
    config: ReproductionConfig,
    selection_path: str | Path,
    *,
    method: str | None = None,
    confirmation_seed: int | None = None,
) -> Path:
    selected = pd.read_csv(selection_path)
    frozen = apply_selected_configuration(
        config,
        selected,
        method=method,
        confirmation_seed=confirmation_seed,
    )
    run_dir = run_reproduction(frozen)
    used = selected[selected["method"] == frozen.experiment.methods[0]]
    used.to_csv(run_dir / "confirmation_selection.csv", index=False)
    (run_dir / "confirmation_provenance.json").write_text(
        json.dumps(
            {
                "selection_path": str(Path(selection_path).resolve()),
                "confirmation_seed": frozen.seed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return run_dir
