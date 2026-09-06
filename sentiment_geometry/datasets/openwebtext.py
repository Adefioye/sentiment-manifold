"""Lazy OpenWebText access so ordinary imports do not download data."""

from __future__ import annotations


def load_openwebtext(
    dataset_name: str = "stas/openwebtext-10k",
    split: str = "train",
    max_samples: int | None = None,
) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    column = "text" if "text" in dataset.column_names else dataset.column_names[0]
    return [str(text) for text in dataset[column] if str(text).strip()]
