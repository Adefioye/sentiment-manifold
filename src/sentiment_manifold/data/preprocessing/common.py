"""Common binary-dataset preprocessing and tokenizer-aware pairing.

These helpers deliberately do not run language models. They construct prompts,
record exact tokenizer lengths, and make deterministic opposite-label pairs.
Model scoring and interventions remain experiment concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from transformers import AutoTokenizer


DEFAULT_PROMPT_TEMPLATE = "Review Text: {text} Review Sentiment:"


@dataclass(frozen=True)
class PairingModelSpec:
    alias: str
    hub_name: str
    column_prefix: str
    prepend_bos: bool
    context_length: int


PAIRING_MODEL_SPECS: dict[str, PairingModelSpec] = {
    "gpt2-small": PairingModelSpec("gpt2-small", "gpt2", "gpt2_small", True, 1024),
    "qwen-0.6b": PairingModelSpec(
        "qwen-0.6b", "Qwen/Qwen3-0.6B-Base", "qwen_0_6b", False, 32768
    ),
    "gemma-2b": PairingModelSpec(
        "gemma-2b", "google/gemma-2b", "gemma_2b", True, 8192
    ),
    "pythia-1.4b": PairingModelSpec(
        "pythia-1.4b", "EleutherAI/pythia-1.4b", "pythia_1_4b", True, 2048
    ),
}
DEFAULT_PAIRING_MODELS = tuple(PAIRING_MODEL_SPECS)


@dataclass(frozen=True)
class BinaryPreprocessingResult:
    output_dir: Path
    datasets: dict[str, DatasetDict]
    metadata: dict[str, Any]
    hub_repo_id: str | None = None


def resolve_pairing_models(models: Sequence[str] | None) -> tuple[PairingModelSpec, ...]:
    """Resolve repeatable CLI selections; no selection and ``all`` mean all four."""
    requested = list(models or DEFAULT_PAIRING_MODELS)
    if "all" in requested:
        requested = list(DEFAULT_PAIRING_MODELS)
    unknown = sorted(set(requested) - set(PAIRING_MODEL_SPECS))
    if unknown:
        raise ValueError(
            f"Unknown pairing models {unknown}; choose from {list(PAIRING_MODEL_SPECS)}"
        )
    return tuple(PAIRING_MODEL_SPECS[name] for name in dict.fromkeys(requested))


def parse_revision_overrides(values: Sequence[str] | None) -> dict[str, str]:
    """Parse repeatable ``MODEL=REVISION`` values."""
    revisions: dict[str, str] = {}
    for value in values or ():
        if "=" not in value:
            raise ValueError(f"Invalid tokenizer revision {value!r}; expected MODEL=REVISION")
        alias, revision = value.split("=", 1)
        if alias not in PAIRING_MODEL_SPECS or not revision:
            raise ValueError(f"Invalid tokenizer revision {value!r}")
        revisions[alias] = revision
    return revisions


def _token_ids(tokenizer, text: str, *, prepend_bos: bool) -> list[int]:
    ids = list(tokenizer(text, add_special_tokens=not prepend_bos)["input_ids"])
    if prepend_bos:
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is None:
            raise ValueError(
                f"{type(tokenizer).__name__} has no BOS token but preprocessing requires one"
            )
        ids = [int(bos_token_id), *map(int, ids)]
    return [int(token_id) for token_id in ids]


def load_pairing_tokenizers(
    specs: Sequence[PairingModelSpec],
    revisions: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load tokenizers only and return immutable provenance where available."""
    revisions = revisions or {}
    tokenizers: dict[str, Any] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for spec in specs:
        requested_revision = revisions.get(spec.alias)
        tokenizer = AutoTokenizer.from_pretrained(
            spec.hub_name,
            revision=requested_revision,
            use_fast=True,
        )
        tokenizers[spec.alias] = tokenizer
        provenance[spec.alias] = {
            "hub_name": spec.hub_name,
            "requested_revision": requested_revision,
            "resolved_revision": tokenizer.init_kwargs.get("_commit_hash"),
            "tokenizer_class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "prepend_bos": spec.prepend_bos,
            "bos_token_id": tokenizer.bos_token_id,
            "context_length": spec.context_length,
        }
    return tokenizers, provenance


def annotate_token_lengths(
    rows: Iterable[dict[str, Any]],
    *,
    specs: Sequence[PairingModelSpec],
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    revisions: Mapping[str, str] | None = None,
    tokenizers: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Add the exact raw/prompt lengths used for model-specific pairing."""
    if "{text}" not in prompt_template:
        raise ValueError("prompt_template must contain the literal placeholder '{text}'")
    if tokenizers is None:
        loaded, provenance = load_pairing_tokenizers(specs, revisions)
    else:
        loaded = dict(tokenizers)
        missing = [spec.alias for spec in specs if spec.alias not in loaded]
        if missing:
            raise ValueError(f"Missing supplied tokenizers for {missing}")
        provenance = {
            spec.alias: {
                "hub_name": spec.hub_name,
                "requested_revision": (revisions or {}).get(spec.alias),
                "resolved_revision": getattr(loaded[spec.alias], "init_kwargs", {}).get(
                    "_commit_hash"
                ),
                "tokenizer_class": type(loaded[spec.alias]).__name__,
                "vocab_size": len(loaded[spec.alias]),
                "prepend_bos": spec.prepend_bos,
                "bos_token_id": loaded[spec.alias].bos_token_id,
                "context_length": spec.context_length,
            }
            for spec in specs
        }

    annotated: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        text = str(row["text"])
        prompt = prompt_template.format(text=text)
        row["prompt"] = prompt
        row["prompt_template"] = prompt_template
        for spec in specs:
            tokenizer = loaded[spec.alias]
            row[f"{spec.column_prefix}_raw_num_tokens"] = len(
                _token_ids(tokenizer, text, prepend_bos=spec.prepend_bos)
            )
            prompt_length = len(_token_ids(tokenizer, prompt, prepend_bos=spec.prepend_bos))
            row[f"{spec.column_prefix}_prompt_num_tokens"] = prompt_length
            row[f"{spec.column_prefix}_fits_context"] = prompt_length <= spec.context_length
        annotated.append(row)
    return annotated, provenance


def _score(row: Mapping[str, Any]) -> float | None:
    for field in ("continuous_score", "sentiment_score", "original_valence_class"):
        value = row.get(field)
        if value is not None:
            return float(value)
    return None


def make_equal_length_matches(
    rows: Iterable[dict[str, Any]],
    *,
    dataset_name: str,
    specs: Sequence[PairingModelSpec],
    pairing_model: str,
    splits: Sequence[str],
) -> list[dict[str, Any]]:
    """Make maximal deterministic pairs with equal full-prompt token lengths.

    ``pairing_model='common'`` requires equality under every selected tokenizer.
    Every example is used at most once per model configuration and split.
    """
    if pairing_model == "common":
        active_specs = tuple(specs)
        if not active_specs:
            raise ValueError("Common pairing requires at least one tokenizer")
    else:
        active_specs = tuple(spec for spec in specs if spec.alias == pairing_model)
        if len(active_specs) != 1:
            raise ValueError(f"Pairing model {pairing_model!r} was not tokenized")

    allowed_splits = set(splits)
    buckets: dict[tuple[str, tuple[int, ...]], dict[int, list[dict[str, Any]]]] = {}
    for source in rows:
        row = dict(source)
        split = str(row["split"])
        if split not in allowed_splits:
            continue
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"Binary pairing requires labels 0/1, got {label}")
        if not all(bool(row[f"{spec.column_prefix}_fits_context"]) for spec in active_specs):
            continue
        signature = tuple(
            int(row[f"{spec.column_prefix}_prompt_num_tokens"]) for spec in active_specs
        )
        buckets.setdefault((split, signature), {0: [], 1: []})[label].append(row)

    matches: list[dict[str, Any]] = []
    for (split, signature), labels in sorted(buckets.items()):
        negative = sorted(labels[0], key=lambda row: str(row["example_id"]))
        positive = sorted(labels[1], key=lambda row: str(row["example_id"]))
        for pos, neg in zip(positive, negative):
            pair_id = (
                f"{dataset_name}-{split}-{pairing_model.replace('.', '_')}-"
                f"pair-{len(matches):06d}"
            )
            match: dict[str, Any] = {
                "pair_id": pair_id,
                "dataset": dataset_name,
                "split": split,
                "pairing_model": pairing_model,
                "positive_example_id": str(pos["example_id"]),
                "positive_text": str(pos["text"]),
                "positive_prompt": str(pos["prompt"]),
                "positive_label": 1,
                "positive_source_score": _score(pos),
                "negative_example_id": str(neg["example_id"]),
                "negative_text": str(neg["text"]),
                "negative_prompt": str(neg["prompt"]),
                "negative_label": 0,
                "negative_source_score": _score(neg),
            }
            for spec, length in zip(active_specs, signature):
                match[f"{spec.column_prefix}_prompt_num_tokens"] = length
            matches.append(match)
    return matches


def make_directed_pairs(matches: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand matches into both receiver-baseline-to-donor-label directions.

    ``source_*`` is the activation donor (the desired label) and ``target_*``
    is the receiver prompt before patching. Thus ``negative_to_positive`` uses
    a positive source and a negative target.
    """
    directed: list[dict[str, Any]] = []
    for match in matches:
        length_fields = {
            key: value for key, value in match.items() if key.endswith("_prompt_num_tokens")
        }
        for target_name, source_name in (("negative", "positive"), ("positive", "negative")):
            source_label = int(source_name == "positive")
            target_label = int(target_name == "positive")
            directed.append(
                {
                    "case_id": f"{match['pair_id']}-{source_name}-to-{target_name}",
                    "pair_id": match["pair_id"],
                    "dataset": match["dataset"],
                    "split": match["split"],
                    "pairing_model": match["pairing_model"],
                    "direction": f"{target_name}_to_{source_name}",
                    "source_example_id": match[f"{source_name}_example_id"],
                    "source_text": match[f"{source_name}_text"],
                    "source_prompt": match[f"{source_name}_prompt"],
                    "source_label": source_label,
                    "source_label_name": source_name,
                    "target_example_id": match[f"{target_name}_example_id"],
                    "target_text": match[f"{target_name}_text"],
                    "target_prompt": match[f"{target_name}_prompt"],
                    "target_label": target_label,
                    "target_label_name": target_name,
                    **length_fields,
                }
            )
    return directed


def dataset_dict_by_split(rows: Sequence[dict[str, Any]]) -> DatasetDict:
    split_order = ("train", "validation", "test")
    present = list(dict.fromkeys([*split_order, *(str(row["split"]) for row in rows)]))
    return DatasetDict(
        {
            split: Dataset.from_list([dict(row) for row in rows if row["split"] == split])
            for split in present
            if any(row["split"] == split for row in rows)
        }
    )


def paired_dataset_dict(rows: Sequence[dict[str, Any]]) -> DatasetDict:
    if not rows:
        return DatasetDict({"test": Dataset.from_list([])})
    return dataset_dict_by_split(rows)


def build_pairing_configs(
    rows: Sequence[dict[str, Any]],
    *,
    dataset_name: str,
    specs: Sequence[PairingModelSpec],
    splits: Sequence[str],
) -> tuple[dict[str, DatasetDict], dict[str, dict[str, int]]]:
    configs: dict[str, DatasetDict] = {"pairing_candidates": dataset_dict_by_split(rows)}
    counts: dict[str, dict[str, int]] = {}
    for name in [*(spec.alias for spec in specs), "common"]:
        matches = make_equal_length_matches(
            rows,
            dataset_name=dataset_name,
            specs=specs,
            pairing_model=name,
            splits=splits,
        )
        directed = make_directed_pairs(matches)
        slug = "common" if name == "common" else PAIRING_MODEL_SPECS[name].column_prefix
        configs[f"{slug}_matched_pairs"] = paired_dataset_dict(matches)
        configs[f"{slug}_directed_pairs"] = paired_dataset_dict(directed)
        counts[name] = {"matched_pairs": len(matches), "directed_pairs": len(directed)}
    return configs, counts


def save_dataset_configs(datasets: Mapping[str, DatasetDict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for config_name, dataset_dict in datasets.items():
        dataset_dict.save_to_disk(output_dir / config_name)


def make_dataset_card(dataset_name: str, metadata: Mapping[str, Any]) -> str:
    configs = "\n".join(f"- config_name: {name}" for name in metadata["dataset_configs"])
    return f"""---
language:
- en
task_categories:
- text-classification
pretty_name: Sentiment Manifold {dataset_name} RQ2 Preprocessing
license: other
configs:
{configs}
---

# Sentiment Manifold {dataset_name} RQ2 Preprocessing

Binary sentiment/valence records plus deterministic equal-full-prompt-length pairs for the
tokenizers declared in `metadata.json`. The `common_*` configurations are equal length under
every selected tokenizer. Labels are dataset ground truth; no language-model prediction is used
to create or relabel examples.

This repository being private controls access but does not replace the source dataset's license,
terms, attribution, or redistribution requirements. Consult the source dataset before sharing.

```json
{json.dumps(dict(metadata), indent=2, sort_keys=True)}
```
"""


def publish_dataset_configs(
    datasets: Mapping[str, DatasetDict],
    *,
    repo_id: str | None,
    default_repo_name: str,
    private: bool,
    card_text: str,
    token: str,
) -> str:
    if not token:
        raise ValueError("A Hugging Face token is required to publish preprocessed datasets")
    api = HfApi(token=token)
    if repo_id is None:
        repo_id = f"{api.whoami()['name']}/{default_repo_name}"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.update_repo_settings(repo_id=repo_id, repo_type="dataset", private=private)
    for config_name, dataset_dict in datasets.items():
        dataset_dict.push_to_hub(
            repo_id,
            config_name=config_name,
            private=private,
            token=token,
        )
    api.upload_file(
        path_or_fileobj=card_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Document RQ2 preprocessing configurations",
    )
    return repo_id


def finish_preprocessing(
    *,
    dataset_name: str,
    output_dir: str | Path,
    datasets: dict[str, DatasetDict],
    metadata: dict[str, Any],
    push_to_hub: bool,
    hub_repo_id: str | None,
    default_repo_name: str,
    private: bool,
    hf_token: str | None,
) -> BinaryPreprocessingResult:
    output_path = Path(output_dir).resolve()
    metadata["dataset_configs"] = list(datasets)
    save_dataset_configs(datasets, output_path)
    card = make_dataset_card(dataset_name, metadata)
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_path / "README.md").write_text(card, encoding="utf-8")
    published_repo = None
    if push_to_hub:
        if not hf_token:
            raise ValueError("Publishing requires HF_TOKEN or HF_TOKEN_PATH")
        published_repo = publish_dataset_configs(
            datasets,
            repo_id=hub_repo_id,
            default_repo_name=default_repo_name,
            private=private,
            card_text=card,
            token=hf_token,
        )
        metadata["hub_repo_id"] = published_repo
        (output_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return BinaryPreprocessingResult(output_path, datasets, metadata, published_repo)
