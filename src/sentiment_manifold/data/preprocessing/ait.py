"""SemEval-2018 Task 1 Affect in Tweets (AIT) valence preprocessing."""

from __future__ import annotations

import csv
from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .common import (
    DEFAULT_MAX_PAIRING_PROMPT_TOKENS,
    DEFAULT_PROMPT_TEMPLATE,
    BinaryPreprocessingResult,
    annotate_token_lengths,
    build_pairing_configs,
    dataset_dict_by_split,
    finish_preprocessing,
    parse_revision_overrides,
    resolve_pairing_models,
)


_SPLIT_ALIASES = {"train": "train", "dev": "validation", "validation": "validation", "test": "test"}


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _normalized_keys(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_"): value
        for key, value in row.items()
        if key is not None
    }


def _validate_official_columns(fieldnames: Sequence[str] | None, path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"AIT file has no header: {path}")
    normalized = {
        re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") for name in fieldnames
    }
    required = {"id", "tweet", "affect_dimension", "intensity_class"}
    missing = sorted(required - normalized)
    if missing:
        delimiter_hint = (
            " The official files use a .txt extension but their four columns are "
            "tab-delimited; do not convert tabs to spaces."
            if len(fieldnames) == 1
            else ""
        )
        raise ValueError(
            f"AIT file {path} is missing official columns {missing}; found {fieldnames}."
            f"{delimiter_hint}"
        )


def _ordinal_class(value: Any) -> int:
    match = re.match(r"\s*([+-]?\d+)", str(value))
    if not match:
        raise ValueError(f"Could not parse AIT valence class from {value!r}")
    score = int(match.group(1))
    if score < -3 or score > 3:
        raise ValueError(f"AIT valence class must be in [-3, 3], got {score}")
    return score


def load_ait_binary(files: Mapping[str, str | Path]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load the official tab-delimited Valence-oc ``.txt`` files.

    The source tweet is retained verbatim. The zero class is excluded and the
    original seven-level ordinal annotation is retained so later analyses do
    not mistake it for a continuous measurement.
    """
    rows: list[dict[str, Any]] = []
    checksums: dict[str, str] = {}
    for requested_split, source_path in files.items():
        split = _SPLIT_ALIASES.get(requested_split.lower())
        if split is None:
            raise ValueError(f"Unknown AIT split {requested_split!r}")
        path = Path(source_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"AIT {requested_split} file does not exist: {path}")
        checksums[str(path)] = _file_sha256(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            _validate_official_columns(reader.fieldnames, path)
            for index, source in enumerate(reader):
                normalized = _normalized_keys(source)
                text = normalized.get("tweet")
                example_id = normalized.get("id")
                dimension = str(normalized.get("affect_dimension", "")).strip().lower()
                class_value = normalized.get("intensity_class")
                if text is None or not str(example_id).strip() or class_value is None:
                    raise ValueError(f"AIT file {path} has an incomplete row at line {index + 2}")
                if dimension != "valence":
                    raise ValueError(
                        f"AIT file {path} line {index + 2} has Affect Dimension "
                        f"{dimension!r}, expected 'valence'"
                    )
                ordinal = _ordinal_class(class_value)
                if ordinal == 0:
                    continue
                label = int(ordinal > 0)
                rows.append(
                    {
                        "example_id": f"ait-{split}-{example_id}",
                        "source_example_id": str(example_id),
                        "text": str(text),
                        "label": label,
                        "label_name": "positive" if label else "negative",
                        "split": split,
                        "original_valence_class": ordinal,
                        "annotation_scale": "ordinal_-3_to_3",
                        "binary_policy": "negative=-3..-1; zero=excluded; positive=1..3",
                    }
                )
    return rows, checksums


def discover_ait_files(ait_root: str | Path) -> dict[str, Path]:
    root = Path(ait_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"AIT root does not exist: {root}")
    discovered: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in {".txt", ".tsv"}
            or "valence" not in path.name.lower()
        ):
            continue
        lower = path.name.lower()
        for source_name in ("train", "dev", "test"):
            if source_name in lower and ("gold" in lower or source_name != "test"):
                discovered.setdefault(source_name, path)
    missing = {"train", "dev", "test"} - set(discovered)
    if missing:
        raise FileNotFoundError(
            f"Could not discover AIT Valence-oc files for {sorted(missing)} below {root}; "
            "pass explicit --train-file/--validation-file/--test-file paths"
        )
    return discovered


def preprocess_ait(
    *,
    output_dir: str | Path,
    ait_root: str | Path | None = None,
    files: Mapping[str, str | Path] | None = None,
    pairing_models: Sequence[str] | None = None,
    pairing_revisions: Sequence[str] | None = None,
    pairing_splits: Sequence[str] = ("train", "validation", "test"),
    max_pairing_prompt_tokens: int = DEFAULT_MAX_PAIRING_PROMPT_TOKENS,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
    private: bool = True,
    hf_token: str | None = None,
    tokenizers: Mapping[str, Any] | None = None,
) -> BinaryPreprocessingResult:
    if files is None:
        if ait_root is None:
            raise ValueError("Provide ait_root or explicit AIT split files")
        files = discover_ait_files(ait_root)
    rows, checksums = load_ait_binary(files)
    specs = resolve_pairing_models(pairing_models)
    revisions = parse_revision_overrides(pairing_revisions)
    annotated, tokenizer_metadata = annotate_token_lengths(
        rows,
        specs=specs,
        prompt_template=prompt_template,
        revisions=revisions,
        tokenizers=tokenizers,
    )
    pairing_configs, pairing_counts = build_pairing_configs(
        annotated,
        dataset_name="ait",
        specs=specs,
        splits=pairing_splits,
        max_prompt_tokens=max_pairing_prompt_tokens,
    )
    datasets = {"binary": dataset_dict_by_split(rows), **pairing_configs}
    metadata = {
        "dataset": "SemEval-2018 Task 1 Affect in Tweets, Valence-oc English",
        "label_policy": {"negative": "-3,-2,-1", "excluded": "0", "positive": "1,2,3"},
        "important_measurement_note": (
            "original_valence_class is ordinal, not a continuous interval-scale score"
        ),
        "source_files_sha256": checksums,
        "prompt_template": prompt_template,
        "pairing_splits": list(pairing_splits),
        "max_pairing_prompt_tokens": max_pairing_prompt_tokens,
        "pairing_models": [spec.alias for spec in specs],
        "tokenizers": tokenizer_metadata,
        "counts": {
            "binary_rows": len(rows),
            "by_split": {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "validation", "test")
            },
            "pairing": pairing_counts,
        },
    }
    return finish_preprocessing(
        dataset_name="AIT Valence-oc Binary",
        output_dir=output_dir,
        datasets=datasets,
        metadata=metadata,
        push_to_hub=push_to_hub,
        hub_repo_id=hub_repo_id,
        default_repo_name="sentiment-manifold-ait-valence-binary",
        private=private,
        hf_token=hf_token,
    )
