"""Cross-run geometry for frozen sentiment and valence direction artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..fitting_methods import DirectionArtifact
from ..models import ModelConfig
from ..persistence import RunArtifactStore


@dataclass(frozen=True)
class DirectionAlignmentSource:
    """One source experiment supplying validation-selected directions."""

    name: str
    experiment_name: str
    run_id: str
    fit_position: str
    selection_dataset: str
    selection_metric: str
    evaluation_dataset: str
    expected_domain: str | None = None
    expected_activation_representation: str | None = None

    def run_root(self, storage_root: str | Path) -> Path:
        return Path(storage_root) / self.experiment_name / "runs" / self.run_id


@dataclass
class DirectionAlignmentConfig:
    """Configuration for comparing selected directions from two source domains."""

    output_experiment_name: str
    models: list[ModelConfig]
    sources: list[DirectionAlignmentSource]
    methods: list[str] = field(default_factory=lambda: ["mean_diff", "das"])
    source_path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> DirectionAlignmentConfig:
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        config = cls(
            output_experiment_name=str(raw.get("output_experiment_name", "")),
            models=[ModelConfig(**item) for item in raw.get("models", [])],
            sources=[
                DirectionAlignmentSource(name=str(name), **values)
                for name, values in raw.get("sources", {}).items()
            ],
            methods=list(raw.get("methods", [])),
            source_path=path,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.output_experiment_name:
            raise ValueError("Direction alignment output experiment name cannot be empty")
        if not self.models:
            raise ValueError("Direction alignment requires at least one model")
        model_names = [model.name for model in self.models]
        if len(model_names) != len(set(model_names)):
            raise ValueError("Direction alignment model names must be unique")
        if not self.methods or len(self.methods) != len(set(self.methods)):
            raise ValueError("Direction alignment methods must be non-empty and unique")
        if set(self.methods) != {"mean_diff", "das"}:
            raise ValueError("Direction alignment requires exactly mean_diff and das")
        source_names = [source.name for source in self.sources]
        if set(source_names) != {"sentiment", "valence"} or len(source_names) != 2:
            raise ValueError(
                "Direction alignment requires exactly sentiment and valence sources"
            )
        for source in self.sources:
            for label, value in (
                ("experiment name", source.experiment_name),
                ("run ID", source.run_id),
                ("fit position", source.fit_position),
                ("selection dataset", source.selection_dataset),
                ("selection metric", source.selection_metric),
                ("evaluation dataset", source.evaluation_dataset),
            ):
                if not value:
                    raise ValueError(f"{source.name} source {label} cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_experiment_name": self.output_experiment_name,
            "models": [asdict(model) for model in self.models],
            "sources": {source.name: asdict(source) for source in self.sources},
            "methods": list(self.methods),
            "source_path": str(self.source_path) if self.source_path else None,
        }


@dataclass(frozen=True)
class SelectedDirection:
    representation: str
    model: str
    method: str
    layer: int
    vector: np.ndarray
    checkpoint_path: Path
    selection_dataset: str
    selection_metric: str
    selection_value_percent: float
    model_revision: str | None
    tokenizer_revision: str | None
    orientation_convention: str


@dataclass(frozen=True)
class DirectionAlignmentResult:
    selected_directions: pd.DataFrame
    similarities: pd.DataFrame
    same_method_alignment: pd.DataFrame
    output_dir: Path


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return table


def _checkpoint_candidates(
    source_root: Path,
    selection_row: pd.Series,
    metadata_row: pd.Series,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    relative = metadata_row.get("artifact_relative_path")
    if isinstance(relative, str) and relative.strip():
        candidates.append(source_root / "directions" / relative)
    for value in (
        selection_row.get("direction_checkpoint"),
        metadata_row.get("artifact_path"),
    ):
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value).expanduser()
        candidates.append(path)
        if "directions" in path.parts:
            reverse_index = tuple(reversed(path.parts)).index("directions")
            directions_index = len(path.parts) - reverse_index - 1
            suffix = path.parts[directions_index + 1 :]
            candidates.append(source_root / "directions" / Path(*suffix))
    return tuple(dict.fromkeys(candidates))


def _source_root_with_manifest(
    source: DirectionAlignmentSource,
    storage_root: str | Path,
) -> Path:
    root = source.run_root(storage_root)
    manifest_path = root / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    json.loads(manifest_path.read_text(encoding="utf-8"))
    return root


def _load_source_model_directions(
    *,
    source: DirectionAlignmentSource,
    source_root: Path,
    model: ModelConfig,
    methods: list[str],
) -> list[SelectedDirection]:
    model_root = source_root / "results" / model.name
    selection = _read_csv(
        model_root / "layer_selection.csv",
        {
            "method",
            "fit_position",
            "selected_layer",
            "selection_dataset",
            "selection_metric",
            "selection_value_percent",
        },
    )
    metadata = _read_csv(
        model_root / "direction_metadata.csv",
        {"method", "fit_position", "layer", "artifact_path"},
    )
    selected_metrics = _read_csv(
        model_root / "selected_metrics.csv",
        {"method", "fit_position", "dataset", "layer"},
    )
    for table_name, table in (
        ("selection", selection),
        ("direction metadata", metadata),
        ("selected metrics", selected_metrics),
    ):
        if "model" in table.columns:
            observed = set(table["model"].astype(str))
            if observed != {model.name}:
                raise RuntimeError(
                    f"{source.name} {table_name} contains models {sorted(observed)}, "
                    f"expected {model.name!r}"
                )
    frozen = selection[
        (selection["fit_position"] == source.fit_position)
        & selection["method"].isin(methods)
    ].copy()
    if len(frozen) != len(methods) or set(frozen["method"]) != set(methods):
        raise RuntimeError(
            f"{source.name}/{model.name} does not contain one frozen row per method"
        )
    if set(frozen["selection_dataset"]) != {source.selection_dataset}:
        raise RuntimeError(
            f"{source.name}/{model.name} has unexpected selection-dataset provenance"
        )
    if set(frozen["selection_metric"]) != {source.selection_metric}:
        raise RuntimeError(
            f"{source.name}/{model.name} has unexpected selection-metric provenance"
        )

    loaded: list[SelectedDirection] = []
    order = {method: index for index, method in enumerate(methods)}
    frozen["method_order"] = frozen["method"].map(order)
    for _, row in frozen.sort_values("method_order", kind="stable").iterrows():
        method = str(row["method"])
        layer = int(row["selected_layer"])
        evaluation_rows = selected_metrics[
            (selected_metrics["method"] == method)
            & (selected_metrics["fit_position"] == source.fit_position)
            & (selected_metrics["dataset"] == source.evaluation_dataset)
        ]
        if len(evaluation_rows) != 1 or int(evaluation_rows.iloc[0]["layer"]) != layer:
            raise RuntimeError(
                f"{source.name}/{model.name}/{method} locked evaluation does not "
                f"use selected layer {layer}"
            )
        metadata_rows = metadata[
            (metadata["method"] == method)
            & (metadata["fit_position"] == source.fit_position)
            & (metadata["layer"].astype(int) == layer)
        ]
        if len(metadata_rows) != 1:
            raise RuntimeError(
                f"Expected one metadata row for "
                f"{source.name}/{model.name}/{method}/layer{layer}; "
                f"found {len(metadata_rows)}"
            )
        metadata_row = metadata_rows.iloc[0]
        checkpoint_path = next(
            (
                candidate.resolve()
                for candidate in _checkpoint_candidates(source_root, row, metadata_row)
                if candidate.is_file()
            ),
            None,
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint found for {source.name}/{model.name}/{method}/layer{layer}"
            )
        artifact = DirectionArtifact.load(checkpoint_path)
        vector = np.asarray(artifact.vector, dtype=np.float64)
        if vector.ndim != 1:
            raise RuntimeError(f"Alignment requires a 1D direction: {checkpoint_path}")
        if artifact.method != method or artifact.layer != layer:
            raise RuntimeError(f"Direction identity mismatch: {checkpoint_path}")
        if artifact.model_name != model.hub_name:
            raise RuntimeError(f"Direction model mismatch: {checkpoint_path}")
        if artifact.metadata.get("fit_position") != source.fit_position:
            raise RuntimeError(f"Direction fit-position mismatch: {checkpoint_path}")
        if model.revision and artifact.metadata.get("model_revision") != model.revision:
            raise RuntimeError(f"Direction model-revision mismatch: {checkpoint_path}")
        if (
            source.expected_domain is not None
            and artifact.metadata.get("domain") != source.expected_domain
        ):
            raise RuntimeError(f"Direction domain mismatch: {checkpoint_path}")
        if (
            source.expected_activation_representation is not None
            and artifact.metadata.get("activation_representation")
            != source.expected_activation_representation
        ):
            raise RuntimeError(
                f"Direction activation-representation mismatch: {checkpoint_path}"
            )
        norm = float(np.linalg.norm(vector))
        if not np.isclose(norm, 1.0, atol=1e-5):
            raise RuntimeError(f"Direction is not unit norm: {checkpoint_path}")
        orientation = str(artifact.metadata.get("orientation_convention", ""))
        if not orientation.startswith("negative_to_positive"):
            raise RuntimeError(f"Direction is not positive-oriented: {checkpoint_path}")
        loaded.append(
            SelectedDirection(
                representation=source.name,
                model=model.name,
                method=method,
                layer=layer,
                vector=vector / norm,
                checkpoint_path=checkpoint_path,
                selection_dataset=str(row["selection_dataset"]),
                selection_metric=str(row["selection_metric"]),
                selection_value_percent=float(row["selection_value_percent"]),
                model_revision=artifact.metadata.get("model_revision"),
                tokenizer_revision=artifact.metadata.get("tokenizer_revision"),
                orientation_convention=orientation,
            )
        )
    return loaded


def _selection_rows(directions: list[SelectedDirection]) -> list[dict[str, Any]]:
    return [
        {
            "representation": direction.representation,
            "model": direction.model,
            "method": direction.method,
            "selected_layer": direction.layer,
            "selection_dataset": direction.selection_dataset,
            "selection_metric": direction.selection_metric,
            "selection_value_percent": direction.selection_value_percent,
            "unit_norm": float(np.linalg.norm(direction.vector)),
            "hidden_size": int(direction.vector.shape[0]),
            "orientation_convention": direction.orientation_convention,
            "model_revision": direction.model_revision,
            "tokenizer_revision": direction.tokenizer_revision,
            "checkpoint_path": str(direction.checkpoint_path),
        }
        for direction in directions
    ]


def _cosine_row(
    *,
    comparison: str,
    model: str,
    row: SelectedDirection,
    column: SelectedDirection,
) -> dict[str, Any]:
    if row.vector.shape != column.vector.shape:
        raise RuntimeError(
            f"Direction dimensions differ for {model}: "
            f"{row.vector.shape} != {column.vector.shape}"
        )
    signed = float(np.clip(row.vector @ column.vector, -1.0, 1.0))
    return {
        "comparison": comparison,
        "model": model,
        "row_representation": row.representation,
        "row_method": row.method,
        "row_layer": row.layer,
        "column_representation": column.representation,
        "column_method": column.method,
        "column_layer": column.layer,
        "signed_cosine": signed,
        "absolute_cosine": abs(signed),
    }


def run_direction_alignment_analysis(
    config: DirectionAlignmentConfig,
    *,
    storage_root: str | Path,
    output_dir: str | Path,
) -> DirectionAlignmentResult:
    """Load frozen source artifacts, compute cosines, and persist audit tables."""

    config.validate()
    source_roots = {
        source.name: _source_root_with_manifest(source, storage_root)
        for source in config.sources
    }
    directions: list[SelectedDirection] = []
    for source in config.sources:
        for model in config.models:
            directions.extend(
                _load_source_model_directions(
                    source=source,
                    source_root=source_roots[source.name],
                    model=model,
                    methods=config.methods,
                )
            )
    lookup = {
        (direction.representation, direction.model, direction.method): direction
        for direction in directions
    }
    expected = len(config.sources) * len(config.models) * len(config.methods)
    if len(lookup) != expected:
        raise RuntimeError(f"Expected {expected} unique selected directions; found {len(lookup)}")

    similarity_rows: list[dict[str, Any]] = []
    for model in config.models:
        for representation in ("sentiment", "valence"):
            for row_method in config.methods:
                for column_method in config.methods:
                    similarity_rows.append(
                        _cosine_row(
                            comparison=f"within_{representation}",
                            model=model.name,
                            row=lookup[(representation, model.name, row_method)],
                            column=lookup[(representation, model.name, column_method)],
                        )
                    )
        for valence_method in config.methods:
            for sentiment_method in config.methods:
                similarity_rows.append(
                    _cosine_row(
                        comparison="valence_vs_sentiment",
                        model=model.name,
                        row=lookup[("valence", model.name, valence_method)],
                        column=lookup[("sentiment", model.name, sentiment_method)],
                    )
                )

    selections = pd.DataFrame(_selection_rows(directions))
    similarities = pd.DataFrame(similarity_rows)
    same_method = similarities[
        (similarities["comparison"] == "valence_vs_sentiment")
        & (similarities["row_method"] == similarities["column_method"])
    ].copy()
    same_method = same_method.rename(
        columns={
            "row_method": "method",
            "row_layer": "valence_layer",
            "column_layer": "sentiment_layer",
        }
    )[
        [
            "model",
            "method",
            "valence_layer",
            "sentiment_layer",
            "signed_cosine",
            "absolute_cosine",
        ]
    ].reset_index(drop=True)

    output_dir = Path(output_dir)
    store = RunArtifactStore(output_dir)
    store.write_rows("selected_directions.csv", selections.to_dict("records"))
    store.write_rows("direction_cosines.csv", similarities.to_dict("records"))
    store.write_rows("same_method_cross_alignment.csv", same_method.to_dict("records"))
    store.write_json(
        "resolved_config.json",
        {
            **config.to_dict(),
            "storage_root": str(Path(storage_root).resolve()),
            "source_run_roots": {
                name: str(path.resolve()) for name, path in source_roots.items()
            },
        },
    )
    return DirectionAlignmentResult(
        selected_directions=selections,
        similarities=similarities,
        same_method_alignment=same_method,
        output_dir=output_dir,
    )


__all__ = [
    "DirectionAlignmentConfig",
    "DirectionAlignmentResult",
    "DirectionAlignmentSource",
    "SelectedDirection",
    "run_direction_alignment_analysis",
]
