"""Revision-pinned loading for materialized Hugging Face dataset configurations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class HuggingFaceRows:
    rows: tuple[dict[str, Any], ...]
    requested_revision: str | None
    resolved_revision: str


def load_hf_parquet_rows(
    repo_id: str,
    *,
    config_name: str,
    split: str,
    revision: str | None = None,
    token: str | None = None,
) -> HuggingFaceRows:
    """Load one materialized Parquet split and retain immutable revision provenance."""

    snapshot = Path(
        snapshot_download(
            repo_id,
            repo_type="dataset",
            revision=revision,
            token=token,
            allow_patterns=f"{config_name}/{split}-*.parquet",
            local_files_only=os.environ.get("HF_HUB_OFFLINE", "").lower()
            in {"1", "true", "yes"},
        )
    )
    parquet_files = sorted((snapshot / config_name).glob(f"{split}-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet shards found for {repo_id}/{config_name}:{split}")
    dataset = load_dataset(
        "parquet",
        data_files={split: [str(path) for path in parquet_files]},
        split=split,
    )
    rows = tuple(dict(row) for row in dataset)
    if not rows:
        raise RuntimeError(f"{repo_id}/{config_name}:{split} contains no rows")
    return HuggingFaceRows(
        rows=rows,
        requested_revision=revision,
        resolved_revision=snapshot.name,
    )


__all__ = ["HuggingFaceRows", "load_hf_parquet_rows"]
