"""Fixed-layer real/random-label selectivity experiment orchestration."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ...activations import extract_last_token_activations
from ...evaluation import DirectionalPatchingEvaluator, PatchingResult
from ...models import CausalLMAdapter, clear_device_cache, resolve_device
from ...persistence import RunArtifactStore
from ...probes import BinaryProbe, evaluate_binary_probe
from ...probes.metrics import midpoint_threshold
from .config import (
    DASSelectivityConfig,
    FixedLayerModelConfig,
    FixedLayerSelectivityConfig,
)
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
    fit_selected_real_probe,
    select_logistic_probe,
    select_mlp1_probe,
)


def _labels(rows) -> np.ndarray:
    return np.asarray([row.label for row in rows], dtype=np.int64)


def _run_name() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _float(value: float) -> float | None:
    return None if not np.isfinite(value) else float(value)


def _finite_or_negative_infinity(value: float) -> float:
    return float(value) if np.isfinite(value) else float("-inf")


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    recalls = [float((predictions[labels == label] == label).mean()) for label in (0, 1)]
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

    def _run_directory(self, *, require_run_id: bool = False) -> Path:
        if require_run_id and self.config.output.run_id is None:
            raise ValueError(
                "Staged tuning/final execution requires output.run_id so both stages "
                "resolve the same artifact directory"
            )
        return Path(self.config.output.output_dir) / (self.config.output.run_id or _run_name())

    def tune(self) -> Path:
        """Tune on real validation data and persist a frozen selection, without final fits."""

        run_dir = self._run_directory(require_run_id=True)
        store = RunArtifactStore(run_dir)
        store.write_json("resolved_config.json", self.config.to_dict())
        tables: dict[str, list[dict[str, Any]]] = {
            "sample_manifest": [],
            "pair_manifest": [],
            "tuning_trials": [],
        }
        selections: dict[str, dict[str, dict[str, Any]]] = {}
        runtime_rows: list[dict[str, Any]] = []
        models = tqdm(
            self.config.models,
            desc="Selectivity tuning models",
            leave=True,
            disable=not self.config.progress.enabled,
            unit="model",
        )
        for model in models:
            models.set_postfix(model=model.name)
            adapter = CausalLMAdapter.from_pretrained(
                model.hub_name,
                resolve_device(model.device, model.dtype),
                revision=model.revision,
                prepend_bos=model.prepend_bos,
            )
            try:
                selections[model.name] = {}
                for dataset_name in tqdm(
                    self.config.data.datasets,
                    desc=f"{model.name} tuning datasets",
                    leave=self.config.progress.leave_completed,
                    disable=not self.config.progress.enabled,
                    unit="dataset",
                ):
                    dataset_model = self._validated_dataset_model(model, dataset_name, adapter)
                    data = self._prepare_data(dataset_name, dataset_model, adapter)
                    tables["sample_manifest"].extend(data.sample_manifest)
                    tables["pair_manifest"].extend(data.pair_manifest)
                    signature = self._tuning_signature(dataset_model, dataset_name)
                    checkpoint_path = (
                        run_dir
                        / "tuning_checkpoints"
                        / model.name
                        / f"{dataset_name}.json"
                    )
                    if checkpoint_path.is_file():
                        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                        if checkpoint.get("signature") == signature:
                            selections[model.name][data.name] = checkpoint["selection"]
                            tables["tuning_trials"].extend(
                                checkpoint.get("tuning_trials", [])
                            )
                            runtime_rows.append(checkpoint["runtime"])
                            self._persist_tuning_progress(
                                store, tables, runtime_rows, selections
                            )
                            continue
                    activations = self._collect_activations(
                        dataset_model, adapter, data, run_dir, allow_cache_read=True
                    )
                    real_labels = {role: _labels(rows) for role, rows in data.examples.items()}
                    selected_settings: dict[str, Any] = {}
                    first_trial_row = len(tables["tuning_trials"])
                    for method in tqdm(
                        self.config.methods,
                        desc=f"{model.name} {data.name} tune",
                        leave=self.config.progress.leave_completed,
                        disable=not self.config.progress.enabled,
                        unit="method",
                    ):
                        if method == "das":
                            selected = self._tune_das(
                                model=dataset_model,
                                adapter=adapter,
                                data=data,
                                tables=tables,
                            )
                            selected_settings[method] = asdict(selected)
                        else:
                            selected_settings[method] = self._tune_probe(
                                model=dataset_model,
                                data=data,
                                activations=activations,
                                real_labels=real_labels,
                                method=method,
                                tables=tables,
                            )
                    selections[model.name][data.name] = selected_settings
                    runtime = {
                        "phase": "tuning",
                        "model": model.name,
                        "dataset": dataset_name,
                        "layer": dataset_model.layer,
                        **adapter.provenance(),
                    }
                    runtime_rows.append(runtime)
                    checkpoint_store = RunArtifactStore(checkpoint_path.parent)
                    checkpoint_store.write_json(
                        checkpoint_path.name,
                        {
                            "signature": signature,
                            "selection": selected_settings,
                            "tuning_trials": tables["tuning_trials"][first_trial_row:],
                            "runtime": runtime,
                        },
                    )
                    # Commit each completed dataset before starting the next long sweep.
                    self._persist_tuning_progress(store, tables, runtime_rows, selections)
            finally:
                adapter_device = adapter.device_spec.device
                del adapter
                clear_device_cache(adapter_device)
        self._validate_selections(selections)
        self._persist_tuning_progress(store, tables, runtime_rows, selections)
        return run_dir

    def _tuning_signature(
        self, model: FixedLayerModelConfig, dataset_name: str
    ) -> dict[str, Any]:
        """Describe the scientific inputs that make a dataset tuning result reusable."""

        method_configs = {
            "das": asdict(self.config.das),
            "logistic_regression": asdict(self.config.logistic_regression),
            "mlp1": asdict(self.config.mlp1),
        }
        return {
            "schema_version": 1,
            "seed": self.config.seed,
            "model": asdict(model),
            "dataset": dataset_name,
            "data": asdict(self.config.data),
            "methods": list(self.config.methods),
            "method_configs": {
                method: method_configs[method]
                for method in self.config.methods
                if method in method_configs
            },
        }

    @staticmethod
    def _persist_tuning_progress(
        store: RunArtifactStore,
        tables: Mapping[str, Sequence[Mapping[str, Any]]],
        runtime_rows: Sequence[Mapping[str, Any]],
        selections: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> None:
        for filename, rows in tables.items():
            store.write_rows(f"{filename}.csv", rows)
        store.write_rows("tuning_runtime.csv", runtime_rows)
        store.write_json("selected_hyperparameters.json", selections)

    def run_selected(
        self,
        selections: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    ) -> Path:
        """Run paired real/random tasks using previously tuned frozen settings."""

        run_dir = self._run_directory(require_run_id=True)
        if selections is None:
            selection_path = run_dir / "selected_hyperparameters.json"
            if not selection_path.is_file():
                raise FileNotFoundError(f"Run validation tuning first; missing {selection_path}")
            selections = json.loads(selection_path.read_text(encoding="utf-8"))
        self._validate_selections(selections)
        store = RunArtifactStore(run_dir)
        store.write_json("resolved_config.json", self.config.to_dict())
        store.write_json(
            "confirmed_hyperparameters.json",
            {model: dict(datasets) for model, datasets in selections.items()},
        )
        tables: dict[str, list[dict[str, Any]]] = {
            "metrics": [],
            "predictions": [],
            "causal_metrics": [],
            "patching_records": [],
            "random_label_assignments": [],
            "random_pair_manifest": [],
            "sample_manifest": [],
            "pair_manifest": [],
            "fit_metadata": [],
            "training_history": [],
        }
        runtime_rows: list[dict[str, Any]] = []
        models = tqdm(
            self.config.models,
            desc="Selectivity final models",
            leave=True,
            disable=not self.config.progress.enabled,
            unit="model",
        )
        for model in models:
            models.set_postfix(model=model.name)
            adapter = CausalLMAdapter.from_pretrained(
                model.hub_name,
                resolve_device(model.device, model.dtype),
                revision=model.revision,
                prepend_bos=model.prepend_bos,
            )
            try:
                for dataset_name in tqdm(
                    self.config.data.datasets,
                    desc=f"{model.name} final datasets",
                    leave=self.config.progress.leave_completed,
                    disable=not self.config.progress.enabled,
                    unit="dataset",
                ):
                    dataset_model = self._validated_dataset_model(model, dataset_name, adapter)
                    data = self._prepare_data(dataset_name, dataset_model, adapter)
                    activations = self._collect_activations(
                        dataset_model, adapter, data, run_dir, allow_cache_read=True
                    )
                    tables["sample_manifest"].extend(data.sample_manifest)
                    tables["pair_manifest"].extend(data.pair_manifest)
                    self._run_dataset_with_selected_settings(
                        dataset_model,
                        adapter,
                        data,
                        activations,
                        run_dir,
                        tables,
                        selections[model.name][data.name],
                    )
                    runtime_rows.append(
                        {
                            "phase": "final",
                            "model": model.name,
                            "dataset": dataset_name,
                            "layer": dataset_model.layer,
                            **adapter.provenance(),
                        }
                    )
            finally:
                adapter_device = adapter.device_spec.device
                del adapter
                clear_device_cache(adapter_device)
        for filename, rows in tables.items():
            if rows or filename != "predictions":
                store.write_rows(f"{filename}.csv", rows)
        store.write_rows("runtime.csv", runtime_rows)
        store.write_rows("selectivity_summary.csv", self._selectivity_summary(tables["metrics"]))
        return run_dir

    def tune_das_epoch_budgets(
        self,
        learning_rate_selections: Mapping[str, Mapping[str, Mapping[str, Any]]],
        epoch_budgets: Sequence[int],
    ) -> Path:
        """Tune DAS epoch budgets without overwriting learning-rate tuning artifacts."""

        if self.config.methods != ["das"]:
            raise ValueError("DAS epoch-budget tuning requires config.methods == ['das']")
        self._validate_selections(learning_rate_selections)
        budgets = tuple(int(value) for value in epoch_budgets)
        if not budgets or any(value < 1 for value in budgets):
            raise ValueError("DAS epoch budgets must be positive integers")
        if len(set(budgets)) != len(budgets):
            raise ValueError("DAS epoch budgets must be unique")

        run_dir = self._run_directory(require_run_id=True)
        store = RunArtifactStore(run_dir)
        store.write_json(
            "das_epoch_tuning_config.json",
            {"epoch_budgets": list(budgets), "selection_metric": "validation_iia"},
        )
        selections = json.loads(json.dumps(learning_rate_selections))
        trial_rows: list[dict[str, Any]] = []
        runtime_rows: list[dict[str, Any]] = []
        models = tqdm(
            self.config.models,
            desc="DAS epoch-budget tuning models",
            leave=True,
            disable=not self.config.progress.enabled,
            unit="model",
        )
        for model in models:
            models.set_postfix(model=model.name)
            adapter = CausalLMAdapter.from_pretrained(
                model.hub_name,
                resolve_device(model.device, model.dtype),
                revision=model.revision,
                prepend_bos=model.prepend_bos,
            )
            try:
                for dataset_name in tqdm(
                    self.config.data.datasets,
                    desc=f"{model.name} DAS epoch datasets",
                    leave=self.config.progress.leave_completed,
                    disable=not self.config.progress.enabled,
                    unit="dataset",
                ):
                    dataset_model = self._validated_dataset_model(model, dataset_name, adapter)
                    data = self._prepare_data(dataset_name, dataset_model, adapter)
                    base_settings = dict(learning_rate_selections[model.name][data.name]["das"])
                    signature = {
                        "schema_version": 1,
                        "tuning": self._tuning_signature(dataset_model, dataset_name),
                        "learning_rate_selection": base_settings,
                        "epoch_budgets": list(budgets),
                    }
                    checkpoint_path = (
                        run_dir
                        / "das_epoch_tuning_checkpoints"
                        / model.name
                        / f"{dataset_name}.json"
                    )
                    if checkpoint_path.is_file():
                        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                        if checkpoint.get("signature") == signature:
                            selections[model.name][data.name]["das"] = checkpoint["selection"]
                            trial_rows.extend(checkpoint.get("tuning_trials", []))
                            runtime_rows.append(checkpoint["runtime"])
                            self._persist_das_epoch_tuning_progress(
                                store, trial_rows, runtime_rows, selections
                            )
                            continue
                    base_config = DASSelectivityConfig(**base_settings)
                    candidates = tuple(
                        replace(base_config, epochs=budget, trials=[]) for budget in budgets
                    )
                    first_trial_row = len(trial_rows)
                    selected = self._tune_das(
                        model=dataset_model,
                        adapter=adapter,
                        data=data,
                        tables={"das_epoch_tuning_trials": trial_rows},
                        candidates=candidates,
                        table_name="das_epoch_tuning_trials",
                        progress_stage="DAS epoch tune",
                    )
                    selections[model.name][data.name]["das"] = asdict(selected)
                    runtime = {
                        "phase": "das_epoch_tuning",
                        "model": model.name,
                        "dataset": dataset_name,
                        "layer": dataset_model.layer,
                        **adapter.provenance(),
                    }
                    runtime_rows.append(runtime)
                    checkpoint_store = RunArtifactStore(checkpoint_path.parent)
                    checkpoint_store.write_json(
                        checkpoint_path.name,
                        {
                            "signature": signature,
                            "selection": asdict(selected),
                            "tuning_trials": trial_rows[first_trial_row:],
                            "runtime": runtime,
                        },
                    )
                    # Persist after each dataset so a completed dataset survives a later failure.
                    self._persist_das_epoch_tuning_progress(
                        store, trial_rows, runtime_rows, selections
                    )
            finally:
                adapter_device = adapter.device_spec.device
                del adapter
                clear_device_cache(adapter_device)
        self._validate_selections(selections)
        self._persist_das_epoch_tuning_progress(
            store, trial_rows, runtime_rows, selections
        )
        return run_dir

    @staticmethod
    def _persist_das_epoch_tuning_progress(
        store: RunArtifactStore,
        trial_rows: Sequence[Mapping[str, Any]],
        runtime_rows: Sequence[Mapping[str, Any]],
        selections: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> None:
        store.write_rows("das_epoch_tuning_trials.csv", trial_rows)
        store.write_rows("das_epoch_tuning_runtime.csv", runtime_rows)
        store.write_json("das_epoch_selected_hyperparameters.json", selections)

    def _validated_dataset_model(
        self,
        model: FixedLayerModelConfig,
        dataset_name: str,
        adapter: CausalLMAdapter,
    ) -> FixedLayerModelConfig:
        layer = model.layer_for(dataset_name)
        if layer > adapter.n_layers:
            raise ValueError(
                f"{model.name}/{dataset_name} layer {layer} exceeds boundary {adapter.n_layers}"
            )
        return replace(model, layer=layer)

    def _activation_cache_path(
        self, run_dir: Path, model: FixedLayerModelConfig, dataset: str
    ) -> Path:
        assert model.layer is not None
        return run_dir / "activations" / model.name / f"{dataset}-layer{model.layer:02d}.npz"

    def _collect_activations(
        self,
        model: FixedLayerModelConfig,
        adapter: CausalLMAdapter,
        data: PreparedSelectivityData,
        run_dir: Path,
        *,
        allow_cache_read: bool,
    ) -> dict[str, np.ndarray]:
        assert model.layer is not None
        cache_path = self._activation_cache_path(run_dir, model, data.name)
        if allow_cache_read and cache_path.is_file():
            with np.load(cache_path) as cached:
                for role, rows in data.examples.items():
                    expected_ids = np.asarray([row.example_id for row in rows])
                    cached_ids = cached[f"{role}_example_ids"]
                    if not np.array_equal(cached_ids, expected_ids):
                        raise ValueError(
                            f"Cached activation IDs do not match {model.name}/{data.name}/{role}"
                        )
                return {role: np.asarray(cached[f"{role}_activations"]) for role in data.examples}
        activations = {
            role: extract_last_token_activations(
                adapter,
                rows,
                model.layer,
                batch_size=model.batch_size,
                show_progress=self.config.progress.enabled,
                progress_description=f"{model.name} {data.name} {role} activations",
                progress_leave=self.config.progress.leave_completed,
            )
            for role, rows in data.examples.items()
        }
        if self.config.output.cache_activations:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                **{f"{role}_activations": values for role, values in activations.items()},
                **{
                    f"{role}_example_ids": np.asarray(
                        [row.example_id for row in data.examples[role]]
                    )
                    for role in data.examples
                },
            )
        return activations

    def _validate_selections(
        self, selections: Mapping[str, Mapping[str, Mapping[str, Any]]]
    ) -> None:
        missing = [
            f"{model.name}/{dataset}/{method}"
            for model in self.config.models
            for dataset in self.config.data.datasets
            for method in self.config.methods
            if model.name not in selections
            or dataset not in selections[model.name]
            or method not in selections[model.name][dataset]
        ]
        if missing:
            raise ValueError(f"Missing frozen hyperparameter selections: {missing}")

    def run(self) -> Path:
        run_dir = Path(self.config.output.output_dir) / (self.config.output.run_id or _run_name())
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
            "tuning_trials": [],
            "training_history": [],
        }
        runtime_rows: list[dict[str, Any]] = []
        models = tqdm(
            self.config.models,
            desc="Selectivity models",
            leave=True,
            disable=not self.config.progress.enabled,
            unit="model",
        )
        for model in models:
            models.set_postfix(model=model.name)
            adapter = CausalLMAdapter.from_pretrained(
                model.hub_name,
                resolve_device(model.device, model.dtype),
                revision=model.revision,
                prepend_bos=model.prepend_bos,
            )
            datasets = tqdm(
                self.config.data.datasets,
                desc=f"{model.name} datasets",
                leave=self.config.progress.leave_completed,
                disable=not self.config.progress.enabled,
                unit="dataset",
            )
            for dataset_name in datasets:
                datasets.set_postfix(dataset=dataset_name)
                layer = model.layer_for(dataset_name)
                if layer > adapter.n_layers:
                    raise ValueError(
                        f"{model.name}/{dataset_name} layer {layer} exceeds boundary "
                        f"{adapter.n_layers}"
                    )
                dataset_model = replace(model, layer=layer)
                runtime_rows.append(
                    {
                        "model": model.name,
                        "dataset": dataset_name,
                        "layer": layer,
                        **adapter.provenance(),
                    }
                )
                data = self._prepare_data(dataset_name, dataset_model, adapter)
                self._run_dataset(dataset_model, adapter, data, run_dir, all_tables)
            adapter_device = adapter.device_spec.device
            del adapter
            clear_device_cache(adapter_device)
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
        activations: dict[str, np.ndarray] = {}
        for role, rows in data.examples.items():
            activations[role] = extract_last_token_activations(
                adapter,
                rows,
                model.layer,
                batch_size=model.batch_size,
                show_progress=self.config.progress.enabled,
                progress_description=(f"{model.name} {data.name} {role} activations"),
                progress_leave=self.config.progress.leave_completed,
            )
        if self.config.output.cache_activations:
            cache_dir = run_dir / "activations" / model.name
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_dir / f"{data.name}-layer{model.layer:02d}.npz",
                **{f"{role}_activations": values for role, values in activations.items()},
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
        selected_settings: dict[str, Any] = {}
        methods = tqdm(
            self.config.methods,
            desc=f"{model.name} {data.name} tune",
            leave=self.config.progress.leave_completed,
            disable=not self.config.progress.enabled,
            unit="method",
        )
        for method in methods:
            methods.set_postfix(method=method)
            if method == "das":
                selected_settings[method] = self._tune_das(
                    model=model,
                    adapter=adapter,
                    data=data,
                    tables=tables,
                )
            else:
                selected_settings[method] = self._tune_probe(
                    model=model,
                    data=data,
                    activations=activations,
                    real_labels=real_labels,
                    method=method,
                    tables=tables,
                )
        seeds = tqdm(
            self.config.random_labels.seeds,
            desc=f"{model.name} {data.name} paired seeds",
            leave=self.config.progress.leave_completed,
            disable=not self.config.progress.enabled,
            unit="seed",
        )
        for run_seed in seeds:
            seeds.set_postfix(seed=run_seed)
            randomized = randomize_data(data, seed=run_seed)
            random_labels = {role: _labels(rows) for role, rows in randomized.examples.items()}
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
                        selected_config=selected_settings[method],
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
                        hyperparameters=selected_settings[method],
                    )

    def _run_dataset_with_selected_settings(
        self,
        model: FixedLayerModelConfig,
        adapter: CausalLMAdapter,
        data: PreparedSelectivityData,
        activations: Mapping[str, np.ndarray],
        run_dir: Path,
        tables: dict[str, list[dict[str, Any]]],
        selected_settings: Mapping[str, Any],
    ) -> None:
        """Run final paired fits only; settings must already be validation-selected."""

        real_labels = {role: _labels(rows) for role, rows in data.examples.items()}
        seeds = tqdm(
            self.config.random_labels.seeds,
            desc=f"{model.name} {data.name} final paired seeds",
            leave=self.config.progress.leave_completed,
            disable=not self.config.progress.enabled,
            unit="seed",
        )
        for run_seed in seeds:
            seeds.set_postfix(seed=run_seed)
            randomized = randomize_data(data, seed=run_seed)
            random_labels = {role: _labels(rows) for role, rows in randomized.examples.items()}
            tables["random_label_assignments"].extend(
                {"model": model.name, "layer": model.layer, **row}
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
                settings = selected_settings[method]
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
                        selected_config=DASSelectivityConfig(**dict(settings)),
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
                        hyperparameters=dict(settings),
                    )

    def _tune_probe(
        self,
        *,
        model: FixedLayerModelConfig,
        data: PreparedSelectivityData,
        method: str,
        activations: Mapping[str, np.ndarray],
        real_labels: Mapping[str, np.ndarray],
        tables: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        if method == "mean_diff":
            return {}
        if method == "logistic_regression":
            selected = select_logistic_probe(
                activations["train"],
                real_labels["train"],
                activations["validation"],
                real_labels["validation"],
                search=self.config.logistic_regression,
                seed=self.config.seed,
            )
        elif method == "mlp1":
            selected = select_mlp1_probe(
                activations["train"],
                real_labels["train"],
                activations["validation"],
                real_labels["validation"],
                search=self.config.mlp1,
                seed=self.config.seed,
                show_progress=self.config.progress.enabled,
                progress_description=f"{model.name} {data.name} MLP-1 tune",
                progress_leave=self.config.progress.leave_completed,
            )
        else:
            raise ValueError(f"Unsupported probe method: {method}")
        tables["tuning_trials"].extend(
            {
                "model": model.name,
                "dataset": data.name,
                "layer": model.layer,
                "method": method,
                "task": "real",
                "seed": self.config.seed,
                **trial,
                "hyperparameters": json.dumps(trial["hyperparameters"], sort_keys=True),
            }
            for trial in selected.tuning_trials
        )
        return selected.hyperparameters

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
        hyperparameters: Mapping[str, Any],
    ) -> None:
        real_probe = fit_selected_real_probe(
            method,
            activations["train"],
            real_labels["train"],
            activations["validation"],
            real_labels["validation"],
            hyperparameters=hyperparameters,
            seed=run_seed,
            show_progress=self.config.progress.enabled,
            progress_description=(f"{model.name} {data.name} {method} real seed {run_seed}"),
            progress_leave=self.config.progress.leave_completed,
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
                int(real_probe.diagnostics["best_epoch"]) + 1 if method == "mlp1" else None
            ),
            show_progress=self.config.progress.enabled,
            progress_description=(f"{model.name} {data.name} {method} random seed {run_seed}"),
            progress_leave=self.config.progress.leave_completed,
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
        state = probe.state()
        selected_epoch = probe.diagnostics.get("best_epoch")
        tables["training_history"].extend(
            {
                "model": model.name,
                "dataset": data.name,
                "layer": model.layer,
                "method": method,
                "task": task,
                "seed": run_seed,
                **row,
                "selected_epoch": int(row["epoch"]) == selected_epoch,
            }
            for row in state.get("history", ())
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

    def _tune_das(
        self,
        *,
        model: FixedLayerModelConfig,
        adapter: CausalLMAdapter,
        data: PreparedSelectivityData,
        tables: dict[str, list[dict[str, Any]]],
        candidates: Sequence[DASSelectivityConfig] | None = None,
        table_name: str = "tuning_trials",
        progress_stage: str = "DAS tune",
    ) -> DASSelectivityConfig:
        assert model.layer is not None
        candidates = tuple(candidates or self.config.das.candidates())
        fitted_candidates = [
            fit_final_token_das(
                adapter,
                data.pairs["train"],
                data.pairs["validation"],
                layer=model.layer,
                answers=data.answers,
                config=candidate,
                seed=self.config.seed,
                evaluation_batch_size=model.batch_size,
                show_progress=self.config.progress.enabled,
                progress_description=(
                    f"{model.name} {data.name} {progress_stage} trial {index + 1}"
                ),
                progress_leave=self.config.progress.leave_completed,
            )
            for index, candidate in enumerate(candidates)
        ]
        selected_index = max(
            range(len(fitted_candidates)),
            key=lambda index: (
                fitted_candidates[index].validation.iia,
                _finite_or_negative_infinity(fitted_candidates[index].validation.recovery),
                -candidates[index].epochs,
            ),
        )
        tables[table_name].extend(
            {
                "model": model.name,
                "dataset": data.name,
                "layer": model.layer,
                "method": "das",
                "task": "real",
                "seed": self.config.seed,
                "trial_index": index,
                "hyperparameters": json.dumps(asdict(candidate), sort_keys=True),
                "validation_native_accuracy": fitted.validation.iia,
                "validation_native_balanced_accuracy": (
                    _causal_balanced_accuracy(fitted.validation)
                ),
                "validation_midpoint_accuracy": None,
                "validation_midpoint_balanced_accuracy": None,
                "validation_loss": 1.0 - fitted.validation.iia,
                "validation_recovery": _float(fitted.validation.recovery),
                "validation_logit_flip": _float(fitted.validation.flip_rate),
                "validation_sign_flip": fitted.validation.sign_flip_rate,
                "best_epoch": int(fitted.fit_result.diagnostics["selected_epoch"]) + 1,
                "epoch_budget": candidate.epochs,
                "best_epoch_near_budget": (
                    int(fitted.fit_result.diagnostics["selected_epoch"]) + 1
                    >= 0.9 * candidate.epochs
                ),
                "selected": index == selected_index,
            }
            for index, (candidate, fitted) in enumerate(zip(candidates, fitted_candidates))
        )
        return candidates[selected_index]

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
        selected_config: DASSelectivityConfig,
    ) -> None:
        assert model.layer is not None
        real_fitted = fit_final_token_das(
            adapter,
            data.pairs["train"],
            data.pairs["validation"],
            layer=model.layer,
            answers=data.answers,
            config=selected_config,
            seed=run_seed,
            evaluation_batch_size=model.batch_size,
            show_progress=self.config.progress.enabled,
            progress_description=(f"{model.name} {data.name} DAS real seed {run_seed}"),
            progress_leave=self.config.progress.leave_completed,
        )
        paired_epochs = int(real_fitted.fit_result.diagnostics["selected_epoch"]) + 1
        random_fitted = fit_final_token_das(
            adapter,
            randomized.pairs["train"],
            randomized.pairs["validation"],
            layer=model.layer,
            answers=data.answers,
            config=selected_config,
            seed=run_seed,
            evaluation_batch_size=model.batch_size,
            fixed_epochs=paired_epochs,
            show_progress=self.config.progress.enabled,
            progress_description=(f"{model.name} {data.name} DAS random seed {run_seed}"),
            progress_leave=self.config.progress.leave_completed,
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
                midpoint_balanced_accuracy = _balanced_accuracy(labels[role], predictions)
                evaluator = DirectionalPatchingEvaluator(
                    adapter,
                    list(pairs[role]),
                    layer=model.layer,
                    answers=data.answers,
                    position="final",
                    batch_size=model.batch_size,
                    show_progress=self.config.progress.enabled,
                    progress_description=(
                        f"{model.name} {data.name} DAS {task} {role} seed {run_seed}"
                    ),
                    progress_leave=self.config.progress.leave_completed,
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
                diagnostics=np.asarray(json.dumps(fitted.fit_result.diagnostics, sort_keys=True)),
            )
            tables["fit_metadata"].append(
                {
                    "model": model.name,
                    "dataset": data.name,
                    "layer": model.layer,
                    "method": "das",
                    "task": task,
                    "seed": run_seed,
                    "hyperparameters": json.dumps(asdict(selected_config), sort_keys=True),
                    "diagnostics": json.dumps(fitted.fit_result.diagnostics, sort_keys=True),
                }
            )
            tables["training_history"].extend(
                {
                    "model": model.name,
                    "dataset": data.name,
                    "layer": model.layer,
                    "method": "das",
                    "task": task,
                    "seed": run_seed,
                    **row,
                }
                for row in fitted.fit_result.diagnostics.get("loss_history", [])
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
                sem = float(data.std(ddof=1) / np.sqrt(len(data))) if len(data) > 1 else 0.0
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


def tune_fixed_layer_selectivity(config: FixedLayerSelectivityConfig) -> Path:
    """Run and persist validation-only tuning without starting final paired fits."""

    return FixedLayerSelectivityExperiment(config).tune()


def tune_fixed_layer_das_epoch_budgets(
    config: FixedLayerSelectivityConfig,
    learning_rate_selections: Mapping[str, Mapping[str, Mapping[str, Any]]],
    epoch_budgets: Sequence[int],
) -> Path:
    """Tune epoch budgets at each dataset's validation-selected DAS learning rate."""

    return FixedLayerSelectivityExperiment(config).tune_das_epoch_budgets(
        learning_rate_selections, epoch_budgets
    )


def run_fixed_layer_selectivity_with_frozen_hyperparameters(
    config: FixedLayerSelectivityConfig,
    selections: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> Path:
    """Run final paired fits using persisted or explicitly supplied selections."""

    return FixedLayerSelectivityExperiment(config).run_selected(selections)


__all__ = [
    "FixedLayerSelectivityExperiment",
    "run_fixed_layer_selectivity",
    "run_fixed_layer_selectivity_with_frozen_hyperparameters",
    "tune_fixed_layer_das_epoch_budgets",
    "tune_fixed_layer_selectivity",
]
