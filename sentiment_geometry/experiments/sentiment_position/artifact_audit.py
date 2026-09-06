"""Completeness checks for saved sentiment-position directions."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DirectionKey = tuple[str, str, int]


@dataclass(frozen=True)
class DirectionArtifactAudit:
    """Summary of expected, recorded, and durable direction artifacts."""

    summary: pd.DataFrame
    problems: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.problems and bool(self.summary["complete"].all())

    @property
    def expected_total(self) -> int:
        return int(self.summary["expected_artifacts"].sum())

    def require_complete(self) -> None:
        if not self.complete:
            detail = "; ".join(self.problems) or "one or more model audits failed"
            raise RuntimeError(f"Direction artifact audit failed: {detail}")


def audit_direction_artifacts(results_dir: str | Path) -> DirectionArtifactAudit:
    """Audit every configured method × position × layer direction in a completed run."""

    results_dir = Path(results_dir)
    manifest_path = results_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing experiment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = tuple(manifest.get("models", ()))
    if not models:
        raise ValueError(f"Experiment manifest has no completed models: {manifest_path}")

    summaries: list[dict[str, object]] = []
    problems: list[str] = []
    for model in models:
        model_dir = results_dir / model
        config_path = model_dir / "resolved_config.json"
        metadata_path = model_dir / "direction_metadata.csv"
        if not config_path.is_file() or not metadata_path.is_file():
            missing = [str(path) for path in (config_path, metadata_path) if not path.is_file()]
            problems.append(f"{model}: missing result files {missing}")
            summaries.append(
                {
                    "model": model,
                    "expected_artifacts": 0,
                    "recorded_artifacts": 0,
                    "existing_artifacts": 0,
                    "missing_combinations": 0,
                    "unexpected_combinations": 0,
                    "duplicate_records": 0,
                    "missing_files": len(missing),
                    "complete": False,
                }
            )
            continue

        resolved = json.loads(config_path.read_text(encoding="utf-8"))
        methods = tuple(resolved["sweep"]["methods"])
        positions = tuple(resolved["sweep"]["fit_positions"])
        layers = tuple(int(layer) for layer in resolved["resolved_layers"])
        expected: set[DirectionKey] = {
            (position, method, layer)
            for position in positions
            for method in methods
            for layer in layers
        }

        metadata = pd.read_csv(metadata_path)
        required_columns = {"fit_position", "method", "layer", "artifact_path"}
        missing_columns = sorted(required_columns - set(metadata.columns))
        if missing_columns:
            raise ValueError(f"{metadata_path} is missing columns: {missing_columns}")
        keys: list[DirectionKey] = [
            (str(row.fit_position), str(row.method), int(row.layer))
            for row in metadata.itertuples(index=False)
        ]
        counts = Counter(keys)
        recorded = set(keys)
        missing_combinations = expected - recorded
        unexpected_combinations = recorded - expected
        duplicate_records = sum(count - 1 for count in counts.values() if count > 1)
        existing_artifacts = sum(Path(path).is_file() for path in metadata["artifact_path"])
        missing_files = len(metadata) - existing_artifacts
        complete = not (
            missing_combinations
            or unexpected_combinations
            or duplicate_records
            or missing_files
        )
        if not complete:
            problems.append(
                f"{model}: {len(missing_combinations)} missing combinations, "
                f"{len(unexpected_combinations)} unexpected combinations, "
                f"{duplicate_records} duplicate records, {missing_files} missing files"
            )
        summaries.append(
            {
                "model": model,
                "expected_artifacts": len(expected),
                "recorded_artifacts": len(metadata),
                "existing_artifacts": existing_artifacts,
                "missing_combinations": len(missing_combinations),
                "unexpected_combinations": len(unexpected_combinations),
                "duplicate_records": duplicate_records,
                "missing_files": missing_files,
                "complete": complete,
            }
        )

    return DirectionArtifactAudit(pd.DataFrame(summaries), tuple(problems))


__all__ = ["DirectionArtifactAudit", "audit_direction_artifacts"]
