"""Fixed-layer real/random-label selectivity experiment orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from ...activations import extract_last_token_activations
from ...evaluation import DirectionalPatchingEvaluator, PatchingResult
from ...models import CausalLMAdapter, clear_device_cache, resolve_device
from ...persistence import RunArtifactStore
from ...probes import BinaryProbe, evaluate_binary_probe, fit_mean_difference_probe
from ...probes.metrics import midpoint_threshold
from .config import FixedLayerModelConfig, FixedLayerSelectivityConfig
from .datasets import (
    PreparedSelectivityData,
    RandomizedSelectivityData,
    prepare_ait_data,
    prepare_toy_data,
    randomize_data,
)
from .fitting import (
    fit_final_token_das,
    fit_selected_control_probe,
    select_logistic_probe,
    select_mlp1_probe,
)


def _labels(rows) -> np.ndarray:
    return np.asarray([row.label for row in rows], dtype=np.int64)


def _run_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _float(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    recalls = [
        float((predictions[labels == label] == label).mean()) for label in (0, 1)
    ]
    return float(np.mean(recalls))


def _causal_balanced_accuracy(result: PatchingResult) -> float:
    labels = np.asarray([record["clean_label"] for record in result.records], dtype=np.int64)
    correct = np.asarray([record["iia_correct"] for record in result.records], dtype=np.float64)
    return float(np.mean([correct[labels == label].mean() for label in (0, 1)]))


class FixedLayerSelectivityExperiment:
    """Run paired real/random-label probes at one frozen layer per model."""

    def __init__(self, config: FixedLayerSelectivityConfig) -> None:
        config.validate(require_layers=True)
        self.config = config

    def run(self) -> Path:
        run_dir = Path(self.config.output.output_dir) / _run_name()
        store = RunArtifactStore(run_dir)
        store.write_json("resolved_config.json", self.config.to_dict())
        all_tables: dict[str, list[dict[str, Any]]] = {
            "metrics": [],
            "predictions": [],
            "causal_metrics": [],
            "patching_records": [],
            "random_label_assignments": [],
            "random_pair_manifest": [],
            "sample_manifest": [],
            "pair_manifest": [],
            "fit_metadata": [],
        }
        runtime_rows: list[dict[str, Any]] = []
        for model in self.config.models:
            adapter = CausalLMAdapter.from_pretrained(
                model.hub_name,
                resolve_device(model.device, model.dtype),
                revision=model.revision,
                prepend_bos=model.prepend_bos,
            )
            assert model.layer is not None
            if model.layer > adapter.n_layers:
                raise ValueError(
                    f"{model.name} layer {model.layer} exceeds boundary {adapter.n_layers}"
                )
            runtime_rows.append(
                {
                    "model": model.name,
                    "layer": model.layer,
                    **adapter.provenance(),
                }
            )
            for dataset_name in self.config.data.datasets:
                data = self._prepare_data(dataset_name, model, adapter)
                self._run_dataset(model, adapter, data, run_dir, all_tables)
            del adapter
            clear_device_cache()
        for filename, rows in all_tables.items():
            if rows or filename != "predictions":
                store.write_rows(f"{filename}.csv", rows)
        store.write_rows("runtime.csv", runtime_rows)
        summary = self._selectivity_summary(all_tables["metrics"])
        store.write_rows("selectivity_summary.csv", summary)
        return run_dir

    def _prepare_data(
        self,
        dataset_name: str,
        model: FixedLayerModelConfig,
        adapter: CausalLMAdapter,
    ) -> PreparedSelectivityData:
        if dataset_name == "toy_movie_review":
            return prepare_toy_data(self.config, model, adapter)
        if dataset_name == "full_ait":
            return prepare_ait_data(self.config, model)
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _run_dataset(
        self,
        model: FixedLayerModelConfig,
        adapter: CausalLMAdapter,
        data: PreparedSelectivityData,
        run_dir: Path,
        tables: dict[str, list[dict[str, Any]]],
    ) -> None:
        assert model.layer is not None
        activations = {
            role: extract_last_token_activations(
                adapter,
                rows,
                model.layer,
                batch_size=model.batch_size,
            )
            for role, rows in data.examples.items()
        }
        if self.config.output.cache_activations:
            cache_dir = run_dir / "activations" / model.name
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_dir / f"{data.name}-layer{model.layer:02d}.npz",
                **{
                    f"{role}_activations": values
                    for role, values in activations.items()
                },
                **{
                    f"{role}_example_ids": np.asarray(
                        [row.example_id for row in data.examples[role]]
                    )
                    for role in data.examples
                },
            )
        tables["sample_manifest"].extend(data.sample_manifest)
        tables["pair_manifest"].extend(data.pair_manifest)
        real_labels = {role: _labels(rows) for role, rows in data.examples.items()}
        for run_seed in self.config.random_labels.seeds:
            randomized = randomize_data(data, seed=run_seed)
            random_labels = {
                role: _labels(rows) for role, rows in randomized.examples.items()
            }
            tables["random_label_assignments"].extend(
                {
                    "model": model.name,
                    "layer": model.layer,
                    **row,
                }
                for row in randomized.assignment_rows
            )
            tables["random_pair_manifest"].extend(
                {
                    "model": model.name,
                    "dataset": data.name,
                    "layer": model.layer,
                    "seed": run_seed,
                    "role": role,
                    "source_example_id": pair.clean.example_id,
                    "source_random_label": pair.clean.label,
                    "target_example_id": pair.corrupted.example_id,
                    "target_random_label": pair.corrupted.label,
                }
                for role, pairs in randomized.pairs.items()
                for pair in pairs
            )
            for method in self.config.methods:
                if method == "das":
                    self._run_das(
                        model=model,
                        adapter=adapter,
                        data=data,
                        randomized=randomized,
                        activations=activations,
                        real_labels=real_labels,
                        random_labels=random_labels,
                        run_seed=run_seed,
                        run_dir=run_dir,
                        tables=tables,
                    )
                else:
                    self._run_probe(
                        model=model,
                        data=data,
                        activations=activations,
                        real_labels=real_labels,
                        random_labels=random_labels,
                        method=method,
                        run_seed=run_seed,
                        run_dir=run_dir,
                        tables=tables,
                    )

    def _select_real_probe(
        self,
        method: str,
        activations: Mapping[str, np.ndarray],
        labels: Mapping[str, np.ndarray],
        *,
        seed: int,
    ):
        if method == "mean_diff":
            probe = fit_mean_difference_probe(activations["train"], labels["train"])
            return probe, {}
        if method == "logistic_regression":
            selected = select_logistic_probe(
                activations["train"],
                labels["train"],
                activations["validation"],
                labels["validation"],
                search=self.config.logistic_regression,
                seed=seed,
            )
            return selected.probe, selected.hyperparameters
        if method == "mlp1":
            selected = select_mlp1_probe(
                activations["train"],
                labels["train"],
                activations["validation"],
                labels["validation"],
                search=self.config.mlp1,
                seed=seed,
            )
            return selected.probe, selected.hyperparameters
        raise ValueError(f"Unsupported probe method: {method}")

    def _run_probe(
        self,
        *,
        model: FixedLayerModelConfig,
        data: PreparedSelectivityData,
        activations: Mapping[str, np.ndarray],
        real_labels: Mapping[str, np.ndarray],
        random_labels: Mapping[str, np.ndarray],
        method: str,
        run_seed: int,
        run_dir: Path,
        tables: dict[str, list[dict[str, Any]]],
    ) -> None:
        real_probe, hyperparameters = self._select_real_probe(
            method, activations, real_labels, seed=run_seed
        )
        random_probe = fit_selected_control_probe(
            method,
            activations["train"],
            random_labels["train"],
            activations["validation"],
            random_labels["validation"],
            hyperparameters=hyperparameters,
            seed=run_seed,
            training_epochs=(
                int(real_probe.diagnostics["best_epoch"]) + 1
                if method == "mlp1"
                else None
            ),
        )
        for task, probe, labels in (
            ("real", real_probe, real_labels),
            ("random", random_probe, random_labels),
        ):
            self._record_probe_metrics(
                model=model,
                data=data,
                activations=activations,
                labels=labels,
                task=task,
                method=method,
                probe=probe,
                run_seed=run_seed,
                hyperparameters=hyperparameters,
                tables=tables,
            )
            self._save_probe(
                run_dir,
                model=model,
                dataset=data.name,
                method=method,
                task=task,
                seed=run_seed,
                probe=probe,
            )

    def _record_probe_metrics(
        self,
        *,
        model: FixedLayerModelConfig,
        data: PreparedSelectivityData,
        activations: Mapping[str, np.ndarray],
        labels: Mapping[str, np.ndarray],
        task: str,
        method: str,
        probe: BinaryProbe,
        run_seed: int,
        hyperparameters: Mapping[str, Any],
        tables: dict[str, list[dict[str, Any]]],
    ) -> None:
        training_midpoint_scores = probe.midpoint_scores(activations["train"])
        for role in ("train", "validation", "test"):
            evaluation = evaluate_binary_probe(
                training_midpoint_scores=training_midpoint_scores,
                training_labels=labels["train"],
                native_scores=probe.native_scores(activations[role]),
                midpoint_scores=probe.midpoint_scores(activations[role]),
                labels=labels[role],
                native_threshold=probe.native_threshold,
            )
            tables["metrics"].append(
                {
                    "model": model.name,
                    "dataset": data.name,
                    "layer": model.layer,
                    "position": "last_non_padding",
                    "method": method,
                    "task": task,
                    "seed": run_seed,
                    "split": role,
                    "native_accuracy": evaluation.native_accuracy,
                    "native_balanced_accuracy": evaluation.native_balanced_accuracy,
                    "midpoint_accuracy": evaluation.midpoint_accuracy,
                    "midpoint_balanced_accuracy": evaluation.midpoint_balanced_accuracy,
                    "prediction_agreement": evaluation.prediction_agreement,
                    "native_threshold": evaluation.native_threshold,
                    "midpoint_threshold": evaluation.midpoint_threshold,
                    "n_examples": evaluation.n_examples,
                }
            )
            if self.config.output.save_predictions:
                tables["predictions"].extend(
                    {
                        "model": model.name,
                        "dataset": data.name,
                        "layer": model.layer,
                        "method": method,
                        "task": task,
                        "seed": run_seed,
                        "split": role,
                        "example_id": example.example_id,
                        "label": int(label),
                        "native_prediction": int(native),
                        "midpoint_prediction": int(midpoint),
                    }
                    for example, label, native, midpoint in zip(
                        data.examples[role],
                        labels[role],
                        evaluation.native_predictions,
                        evaluation.midpoint_predictions,
                    )
                )
        tables["fit_metadata"].append(
            {
                "model": model.name,
                "dataset": data.name,
                "layer": model.layer,
                "method": method,
                "task": task,
                "seed": run_seed,
                "hyperparameters": json.dumps(dict(hyperparameters), sort_keys=True),
                "diagnostics": json.dumps(probe.diagnostics, sort_keys=True),
            }
        )

    @staticmethod
    def _save_probe(
        run_dir: Path,
        *,
        model: FixedLayerModelConfig,
        dataset: str,
        method: str,
        task: str,
        seed: int,
        probe: BinaryProbe,
    ) -> None:
        directory = run_dir / "checkpoints" / model.name / dataset / method
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(probe.state(), directory / f"{task}-seed{seed}.pt")

    def _run_das(
        self,
        *,
        model: FixedLayerModelConfig,
        adapter: CausalLMAdapter,
        data: PreparedSelectivityData,
        randomized: RandomizedSelectivityData,
        activations: Mapping[str, np.ndarray],
        real_labels: Mapping[str, np.ndarray],
        random_labels: Mapping[str, np.ndarray],
        run_seed: int,
        run_dir: Path,
        tables: dict[str, list[dict[str, Any]]],
    ) -> None:
        assert model.layer is not None
        real_fitted = fit_final_token_das(
            adapter,
            data.pairs["train"],
            data.pairs["validation"],
            layer=model.layer,
            answers=data.answers,
            config=self.config.das,
            seed=run_seed,
            evaluation_batch_size=model.batch_size,
        )
        paired_epochs = int(real_fitted.fit_result.diagnostics["selected_epoch"]) + 1
        random_fitted = fit_final_token_das(
            adapter,
            randomized.pairs["train"],
            randomized.pairs["validation"],
            layer=model.layer,
            answers=data.answers,
            config=self.config.das,
            seed=run_seed,
            evaluation_batch_size=model.batch_size,
            fixed_epochs=paired_epochs,
        )
        for task, pairs, labels, fitted in (
            ("real", data.pairs, real_labels, real_fitted),
            ("random", randomized.pairs, random_labels, random_fitted),
        ):
            direction = fitted.fit_result.direction
            train_scores = activations["train"] @ direction
            threshold = midpoint_threshold(train_scores, labels["train"])
            for role in ("train", "validation", "test"):
                scores = activations[role] @ direction
                predictions = (scores >= threshold).astype(np.int64)
                midpoint_accuracy = float((predictions == labels[role]).mean())
                midpoint_balanced_accuracy = _balanced_accuracy(
                    labels[role], predictions
                )
                evaluator = DirectionalPatchingEvaluator(
                    adapter,
                    list(pairs[role]),
                    layer=model.layer,
                    answers=data.answers,
                    position="final",
                    batch_size=model.batch_size,
                )
                causal = evaluator.evaluate(direction)
                tables["metrics"].append(
                    {
                        "model": model.name,
                        "dataset": data.name,
                        "layer": model.layer,
                        "position": "last_non_padding",
                        "method": "das",
                        "task": task,
                        "seed": run_seed,
                        "split": role,
                        "native_accuracy": causal.iia,
                        "native_balanced_accuracy": _causal_balanced_accuracy(causal),
                        "midpoint_accuracy": midpoint_accuracy,
                        "midpoint_balanced_accuracy": midpoint_balanced_accuracy,
                        "prediction_agreement": None,
                        "native_threshold": None,
                        "midpoint_threshold": threshold,
                        "n_examples": len(labels[role]),
                    }
                )
                self._record_causal(
                    tables,
                    model=model,
                    dataset=data.name,
                    task=task,
                    seed=run_seed,
                    split=role,
                    result=causal,
                )
                if self.config.output.save_predictions:
                    tables["predictions"].extend(
                        {
                            "model": model.name,
                            "dataset": data.name,
                            "layer": model.layer,
                            "method": "das",
                            "task": task,
                            "seed": run_seed,
                            "split": role,
                            "example_id": example.example_id,
                            "label": int(label),
                            "native_prediction": None,
                            "midpoint_prediction": int(prediction),
                        }
                        for example, label, prediction in zip(
                            data.examples[role], labels[role], predictions
                        )
                    )
            checkpoint = (
                run_dir
                / "checkpoints"
                / model.name
                / data.name
                / "das"
                / f"{task}-seed{run_seed}.npz"
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                checkpoint,
                direction=direction,
                diagnostics=np.asarray(
                    json.dumps(fitted.fit_result.diagnostics, sort_keys=True)
                ),
            )
            tables["fit_metadata"].append(
                {
                    "model": model.name,
                    "dataset": data.name,
                    "layer": model.layer,
                    "method": "das",
                    "task": task,
                    "seed": run_seed,
                    "hyperparameters": json.dumps(asdict(self.config.das), sort_keys=True),
                    "diagnostics": json.dumps(
                        fitted.fit_result.diagnostics, sort_keys=True
                    ),
                }
            )

    @staticmethod
    def _record_causal(
        tables: dict[str, list[dict[str, Any]]],
        *,
        model: FixedLayerModelConfig,
        dataset: str,
        task: str,
        seed: int,
        split: str,
        result: PatchingResult,
    ) -> None:
        tables["causal_metrics"].append(
            {
                "model": model.name,
                "dataset": dataset,
                "layer": model.layer,
                "method": "das",
                "task": task,
                "seed": seed,
                "split": split,
                "iia": result.iia,
                "patched_accuracy": result.patched_accuracy,
                "corrupted_accuracy": result.corrupted_accuracy,
                "clean_accuracy": result.clean_accuracy,
                "recovery": _float(result.recovery),
                "recovery_percent": _float(result.recovery_percent),
                "logit_flip": _float(result.flip_rate),
                "logit_flip_percent": _float(result.flip_percent),
                "sign_flip_rate": result.sign_flip_rate,
                "n_pairs": result.n_pairs,
            }
        )
        tables["patching_records"].extend(
            {
                "model": model.name,
                "dataset": dataset,
                "layer": model.layer,
                "method": "das",
                "task": task,
                "seed": seed,
                "split": split,
                **record,
            }
            for record in result.records
        )

    @staticmethod
    def _selectivity_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not rows:
            return []
        frame = pd.DataFrame(rows)
        keys = ["model", "dataset", "layer", "method", "seed", "split"]
        real = frame[frame.task == "real"].set_index(keys)
        random = frame[frame.task == "random"].set_index(keys)
        paired = real.join(random, lsuffix="_real", rsuffix="_random", how="inner")
        records: list[dict[str, Any]] = []
        for metric in (
            "native_accuracy",
            "native_balanced_accuracy",
            "midpoint_accuracy",
            "midpoint_balanced_accuracy",
        ):
            values = paired[f"{metric}_real"] - paired[f"{metric}_random"]
            paired_metric = values.rename("selectivity").reset_index()
            paired_metric["metric"] = metric
            for group, selected in paired_metric.groupby(
                ["model", "dataset", "layer", "method", "split", "metric"],
                dropna=False,
            ):
                model, dataset, layer, method, split, metric_name = group
                data = selected.selectivity.astype(float).to_numpy()
                sem = (
                    float(data.std(ddof=1) / np.sqrt(len(data)))
                    if len(data) > 1
                    else 0.0
                )
                mean = float(data.mean())
                records.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "layer": layer,
                        "method": method,
                        "split": split,
                        "metric": metric_name,
                        "selectivity_mean": mean,
                        "selectivity_std": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
                        "selectivity_sem": sem,
                        "selectivity_ci95_low": mean - 1.96 * sem,
                        "selectivity_ci95_high": mean + 1.96 * sem,
                        "n_seeds": len(data),
                    }
                )
        return records


def run_fixed_layer_selectivity(config: FixedLayerSelectivityConfig) -> Path:
    return FixedLayerSelectivityExperiment(config).run()


__all__ = ["FixedLayerSelectivityExperiment", "run_fixed_layer_selectivity"]
