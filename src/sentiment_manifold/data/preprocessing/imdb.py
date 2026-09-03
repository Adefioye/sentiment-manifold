"""IMDb binary-sentiment preprocessing."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from datasets import DatasetDict, load_dataset

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


def imdb_rows(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize the labeled train/test splits without changing review text."""
    rows: list[dict[str, Any]] = []
    for split_name, records in dataset.items():
        split = "validation" if split_name in {"dev", "validation"} else str(split_name)
        if split not in {"train", "validation", "test"}:
            continue
        for index, source in enumerate(records):
            label = int(source["label"])
            if label not in (0, 1):
                continue
            text = str(source["text"])
            digest = sha256(text.encode("utf-8")).hexdigest()[:16]
            rows.append(
                {
                    "example_id": f"imdb-{split}-{index}-{digest}",
                    "text": text,
                    "label": label,
                    "label_name": "positive" if label else "negative",
                    "split": split,
                    "source_row_index": index,
                }
            )
    return rows


def preprocess_imdb(
    *,
    output_dir: str | Path,
    dataset_name: str = "stanfordnlp/imdb",
    dataset_revision: str | None = None,
    pairing_models: Sequence[str] | None = None,
    pairing_revisions: Sequence[str] | None = None,
    pairing_splits: Sequence[str] = ("test",),
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
    private: bool = True,
    hf_token: str | None = None,
    tokenizers: Mapping[str, Any] | None = None,
    source_dataset: DatasetDict | Mapping[str, Any] | None = None,
) -> BinaryPreprocessingResult:
    source = source_dataset
    if source is None:
        source = load_dataset(dataset_name, revision=dataset_revision)
    rows = imdb_rows(source)
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
        dataset_name="imdb",
        specs=specs,
        splits=pairing_splits,
    )
    datasets = {"binary": dataset_dict_by_split(rows), **pairing_configs}
    metadata = {
        "dataset": dataset_name,
        "requested_dataset_revision": dataset_revision,
        "label_policy": {"negative": 0, "positive": 1},
        "text_policy": "source review text retained verbatim",
        "prompt_template": prompt_template,
        "pairing_splits": list(pairing_splits),
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
        dataset_name="IMDb Binary Sentiment",
        output_dir=output_dir,
        datasets=datasets,
        metadata=metadata,
        push_to_hub=push_to_hub,
        hub_repo_id=hub_repo_id,
        default_repo_name="sentiment-manifold-imdb-binary",
        private=private,
        hf_token=hf_token,
    )
