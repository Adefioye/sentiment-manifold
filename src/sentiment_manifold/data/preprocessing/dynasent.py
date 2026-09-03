"""DynaSent Round 1/2 binary-sentiment preprocessing."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .common import (
    DEFAULT_PROMPT_TEMPLATE,
    BinaryPreprocessingResult,
    annotate_token_lengths,
    build_pairing_configs,
    dataset_dict_by_split,
    finish_preprocessing,
    parse_revision_overrides,
    resolve_pairing_models,
)


def discover_dynasent_files(root: str | Path, rounds: Sequence[int]) -> dict[tuple[int, str], Path]:
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"DynaSent root does not exist: {root_path}")
    discovered: dict[tuple[int, str], Path] = {}
    for path in sorted(root_path.rglob("*.jsonl")):
        lower = path.name.lower()
        round_match = re.search(r"(?:round|r)[-_]?0?([12])", lower)
        if not round_match:
            continue
        round_number = int(round_match.group(1))
        if round_number not in rounds:
            continue
        for source_split, split in (
            ("train", "train"),
            ("dev", "validation"),
            ("validation", "validation"),
            ("test", "test"),
        ):
            if source_split in lower:
                discovered.setdefault((round_number, split), path)
    missing = [
        (round_number, split)
        for round_number in rounds
        for split in ("train", "validation", "test")
        if (round_number, split) not in discovered
    ]
    if missing:
        raise FileNotFoundError(
            f"Could not discover DynaSent JSONL files for {missing} below {root_path}"
        )
    return discovered


def load_dynasent_binary(
    files: Mapping[tuple[int, str], str | Path]
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, str], dict[int, int]]:
    rows_by_round: dict[int, list[dict[str, Any]]] = {}
    checksums: dict[str, str] = {}
    removed_by_round: dict[int, int] = {}
    for (round_number, split), source_path in files.items():
        path = Path(source_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DynaSent source file does not exist: {path}")
        checksums[str(path)] = sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                source = json.loads(line)
                label_name = str(source.get("gold_label", source.get("label", ""))).lower()
                if label_name not in {"positive", "negative"}:
                    removed_by_round[round_number] = removed_by_round.get(round_number, 0) + 1
                    continue
                text = source.get("sentence", source.get("text"))
                if text is None:
                    raise ValueError(f"Missing DynaSent sentence in {path}:{index + 1}")
                source_id = source.get("text_id", source.get("sentence_id", index))
                label = int(label_name == "positive")
                rows_by_round.setdefault(round_number, []).append(
                    {
                        "example_id": f"dynasent-r{round_number}-{split}-{source_id}",
                        "source_example_id": str(source_id),
                        "text": str(text),
                        "label": label,
                        "label_name": label_name,
                        "split": split,
                        "round": round_number,
                        "binary_policy": "gold positive/negative retained; other labels excluded",
                    }
                )
    return rows_by_round, checksums, removed_by_round


def preprocess_dynasent(
    *,
    dynasent_root: str | Path,
    output_dir: str | Path,
    rounds: Sequence[int] = (1, 2),
    files: Mapping[tuple[int, str], str | Path] | None = None,
    pairing_models: Sequence[str] | None = None,
    pairing_revisions: Sequence[str] | None = None,
    pairing_splits: Sequence[str] = ("test",),
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
    private: bool = True,
    hf_token: str | None = None,
    tokenizers: Mapping[str, Any] | None = None,
) -> BinaryPreprocessingResult:
    invalid_rounds = sorted(set(rounds) - {1, 2})
    if invalid_rounds:
        raise ValueError(f"DynaSent rounds must be 1 and/or 2, got {invalid_rounds}")
    source_files = files or discover_dynasent_files(dynasent_root, rounds)
    rows_by_round, checksums, removed = load_dynasent_binary(source_files)
    specs = resolve_pairing_models(pairing_models)
    revisions = parse_revision_overrides(pairing_revisions)
    datasets: dict[str, Any] = {}
    round_metadata: dict[str, Any] = {}
    tokenizer_metadata: dict[str, Any] = {}
    for round_number in rounds:
        rows = rows_by_round.get(round_number, [])
        annotated, tokenizer_metadata = annotate_token_lengths(
            rows,
            specs=specs,
            prompt_template=prompt_template,
            revisions=revisions,
            tokenizers=tokenizers,
        )
        pairing_configs, pairing_counts = build_pairing_configs(
            annotated,
            dataset_name=f"dynasent-r{round_number}",
            specs=specs,
            splits=pairing_splits,
        )
        datasets[f"r{round_number}_binary"] = dataset_dict_by_split(rows)
        datasets.update({f"r{round_number}_{name}": value for name, value in pairing_configs.items()})
        round_metadata[f"r{round_number}"] = {
            "binary_rows": len(rows),
            "excluded_non_binary_rows": removed.get(round_number, 0),
            "by_split": {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "validation", "test")
            },
            "pairing": pairing_counts,
        }
    metadata = {
        "dataset": "DynaSent 1.1 Round 1 and Round 2",
        "rounds": list(rounds),
        "label_policy": "retain gold positive/negative; exclude neutral and non-binary labels",
        "source_files_sha256": checksums,
        "prompt_template": prompt_template,
        "pairing_splits": list(pairing_splits),
        "pairing_models": [spec.alias for spec in specs],
        "tokenizers": tokenizer_metadata,
        "counts": round_metadata,
    }
    return finish_preprocessing(
        dataset_name="DynaSent R1/R2 Binary",
        output_dir=output_dir,
        datasets=datasets,
        metadata=metadata,
        push_to_hub=push_to_hub,
        hub_repo_id=hub_repo_id,
        default_repo_name="sentiment-manifold-dynasent-r1-r2-binary",
        private=private,
        hf_token=hf_token,
    )
