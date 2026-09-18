"""CEBaB binary-sentiment and human counterfactual preprocessing."""

from __future__ import annotations

import json
from ast import literal_eval
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset

from .common import (
    DEFAULT_MAX_PAIRING_PROMPT_TOKENS,
    DEFAULT_PROMPT_TEMPLATE,
    BinaryPreprocessingResult,
    PairingModelSpec,
    annotate_token_lengths,
    build_pairing_configs,
    dataset_dict_by_split,
    finish_preprocessing,
    paired_dataset_dict,
    parse_revision_overrides,
    resolve_pairing_models,
)

ASPECTS = ("food", "ambiance", "service", "noise")
SOURCE_TRAIN_SPLITS = ("train_inclusive", "train_exclusive", "train_observational")
NO_MAJORITY = "no majority"


def cebab_binary_label(review_majority: Any) -> int | None:
    """Apply the authors' binary task: drop 3/no-majority; 1-2=0 and 4-5=1."""
    value = str(review_majority).strip().lower()
    if value in {"", "none", "3", NO_MAJORITY}:
        return None
    try:
        rating = int(value)
    except ValueError as error:
        raise ValueError(f"Invalid CEBaB review_majority value {review_majority!r}") from error
    if rating not in {1, 2, 4, 5}:
        raise ValueError(f"CEBaB review_majority must be 1..5 or 'no majority', got {rating}")
    return int(rating > 3)


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "none"} else text


def _distribution_json(value: Any) -> str:
    """Keep annotator disagreement in a stable Arrow-compatible representation."""
    if value in (None, ""):
        return "{}"
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = literal_eval(value)
    if not isinstance(value, Mapping):
        raise ValueError(f"CEBaB review_label_distribution must be a mapping, got {value!r}")
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _selected_split(source_split: str, train_split: str) -> str | None:
    if source_split == train_split:
        return "train"
    if source_split in {"dev", "validation"}:
        return "validation"
    if source_split == "test":
        return "test"
    return None


def cebab_binary_rows(
    dataset: Mapping[str, Any],
    *,
    train_split: str = "train_inclusive",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize one CEBaB train variant plus validation/test into binary rows."""
    if train_split not in SOURCE_TRAIN_SPLITS:
        raise ValueError(f"train_split must be one of {SOURCE_TRAIN_SPLITS}, got {train_split!r}")
    if train_split not in dataset:
        raise ValueError(f"CEBaB source dataset has no {train_split!r} split")

    rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
    excluded = {"neutral_rating_3": 0, "no_majority": 0}
    for source_split, records in dataset.items():
        split = _selected_split(str(source_split), train_split)
        if split is None:
            continue
        for index, source in enumerate(records):
            source_counts[split] += 1
            majority = str(source["review_majority"])
            label = cebab_binary_label(majority)
            if label is None:
                if majority.strip().lower() == "3":
                    excluded["neutral_rating_3"] += 1
                else:
                    excluded["no_majority"] += 1
                continue

            example_id = str(source.get("id", f"{source_split}-{index}"))
            edit_type = _optional_string(source.get("edit_type"))
            row: dict[str, Any] = {
                "example_id": f"cebab-{example_id}",
                "source_example_id": example_id,
                "original_id": str(source["original_id"]),
                "edit_id": str(source.get("edit_id", "")),
                "text": str(source["description"]),
                "label": label,
                "label_name": "positive" if label else "negative",
                "split": split,
                "source_split": str(source_split),
                "review_rating": int(majority),
                "review_majority": majority,
                "review_label_distribution": _distribution_json(
                    source.get("review_label_distribution")
                ),
                "is_original": bool(source["is_original"]),
                "counterfactual": not bool(source["is_original"]),
                "edit_goal": _optional_string(source.get("edit_goal")),
                "edit_type": edit_type,
                "binary_policy": "1,2=negative; 3/no-majority=excluded; 4,5=positive",
            }
            for aspect in ASPECTS:
                row[f"{aspect}_aspect_majority"] = _optional_string(
                    source.get(f"{aspect}_aspect_majority")
                )
            row["edited_aspect_majority"] = (
                row[f"{edit_type}_aspect_majority"] if edit_type in ASPECTS else None
            )
            rows.append(row)

    metadata = {
        "source_rows": sum(source_counts.values()),
        "source_by_split": source_counts,
        "excluded": excluded,
    }
    return rows, metadata


def _counterfactual_pair(
    original: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
) -> dict[str, Any]:
    edit_goal = counterfactual.get("edit_goal")
    edited_aspect = counterfactual.get("edited_aspect_majority")
    goal_matches = None
    if edit_goal is not None and edited_aspect is not None:
        goal_matches = str(edit_goal).lower() == str(edited_aspect).lower()
    pair: dict[str, Any] = {
        "pair_id": f"cebab-{counterfactual['source_example_id']}",
        "dataset": "cebab",
        "split": counterfactual["split"],
        "original_id": counterfactual["original_id"],
        "edit_id": counterfactual["edit_id"],
        "edit_type": counterfactual.get("edit_type"),
        "edit_goal": edit_goal,
        "edited_aspect_majority": edited_aspect,
        "edit_goal_matches_validated_aspect": goal_matches,
        "original_example_id": original["example_id"],
        "original_text": original["text"],
        "original_prompt": original["prompt"],
        "original_label": int(original["label"]),
        "original_label_name": original["label_name"],
        "original_review_rating": int(original["review_rating"]),
        "counterfactual_example_id": counterfactual["example_id"],
        "counterfactual_text": counterfactual["text"],
        "counterfactual_prompt": counterfactual["prompt"],
        "counterfactual_label": int(counterfactual["label"]),
        "counterfactual_label_name": counterfactual["label_name"],
        "counterfactual_review_rating": int(counterfactual["review_rating"]),
        "rating_delta": int(counterfactual["review_rating"]) - int(original["review_rating"]),
        "polarity_flipped": int(original["label"]) != int(counterfactual["label"]),
    }
    for aspect in ASPECTS:
        field = f"{aspect}_aspect_majority"
        pair[f"original_{field}"] = original.get(field)
        pair[f"counterfactual_{field}"] = counterfactual.get(field)
    for key, value in original.items():
        if key.endswith("_prompt_num_tokens") or key.endswith("_fits_context"):
            pair[f"original_{key}"] = value
    for key, value in counterfactual.items():
        if key.endswith("_prompt_num_tokens") or key.endswith("_fits_context"):
            pair[f"counterfactual_{key}"] = value
    return pair


def make_original_edit_pairs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair every retained human edit with its retained original review."""
    originals: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row["is_original"]:
            key = (str(row["split"]), str(row["original_id"]))
            if key in originals:
                raise ValueError(f"Multiple original CEBaB rows for split/original_id {key}")
            originals[key] = row

    pairs: list[dict[str, Any]] = []
    for counterfactual in sorted(rows, key=lambda row: str(row["example_id"])):
        if counterfactual["is_original"]:
            continue
        key = (str(counterfactual["split"]), str(counterfactual["original_id"]))
        original = originals.get(key)
        if original is not None:
            pairs.append(_counterfactual_pair(original, counterfactual))
    return pairs


def _active_specs(
    specs: Sequence[PairingModelSpec], pairing_model: str
) -> tuple[PairingModelSpec, ...]:
    if pairing_model == "common":
        if not specs:
            raise ValueError("Common pairing requires at least one tokenizer")
        return tuple(specs)
    active = tuple(spec for spec in specs if spec.alias == pairing_model)
    if len(active) != 1:
        raise ValueError(f"Pairing model {pairing_model!r} was not tokenized")
    return active


def make_equal_length_counterfactual_matches(
    rows: Sequence[dict[str, Any]],
    *,
    specs: Sequence[PairingModelSpec],
    pairing_model: str,
    splits: Sequence[str],
    max_prompt_tokens: int = DEFAULT_MAX_PAIRING_PROMPT_TOKENS,
) -> list[dict[str, Any]]:
    """Keep polarity-flipping human edits with equal full-prompt token lengths."""
    active_specs = _active_specs(specs, pairing_model)
    allowed_splits = set(splits)
    matches: list[dict[str, Any]] = []
    for pair in make_original_edit_pairs(rows):
        if pair["split"] not in allowed_splits or not pair["polarity_flipped"]:
            continue
        eligible = True
        lengths: dict[str, int] = {}
        for spec in active_specs:
            prefix = spec.column_prefix
            original_length = int(pair[f"original_{prefix}_prompt_num_tokens"])
            counterfactual_length = int(pair[f"counterfactual_{prefix}_prompt_num_tokens"])
            if (
                not pair[f"original_{prefix}_fits_context"]
                or not pair[f"counterfactual_{prefix}_fits_context"]
                or original_length != counterfactual_length
                or original_length > max_prompt_tokens
            ):
                eligible = False
                break
            lengths[f"{prefix}_prompt_num_tokens"] = original_length
        if not eligible:
            continue

        original_is_positive = int(pair["original_label"]) == 1
        positive_side = "original" if original_is_positive else "counterfactual"
        negative_side = "counterfactual" if original_is_positive else "original"
        matches.append(
            {
                "pair_id": pair["pair_id"],
                "dataset": "cebab",
                "split": pair["split"],
                "pairing_model": pairing_model,
                "original_id": pair["original_id"],
                "edit_id": pair["edit_id"],
                "edit_type": pair["edit_type"],
                "edit_goal": pair["edit_goal"],
                "edit_goal_matches_validated_aspect": pair[
                    "edit_goal_matches_validated_aspect"
                ],
                "rating_delta": pair["rating_delta"],
                "positive_example_id": pair[f"{positive_side}_example_id"],
                "positive_text": pair[f"{positive_side}_text"],
                "positive_prompt": pair[f"{positive_side}_prompt"],
                "positive_label": 1,
                "positive_review_rating": pair[f"{positive_side}_review_rating"],
                "positive_is_original": positive_side == "original",
                "negative_example_id": pair[f"{negative_side}_example_id"],
                "negative_text": pair[f"{negative_side}_text"],
                "negative_prompt": pair[f"{negative_side}_prompt"],
                "negative_label": 0,
                "negative_review_rating": pair[f"{negative_side}_review_rating"],
                "negative_is_original": negative_side == "original",
                **lengths,
            }
        )
    return matches


def make_counterfactual_directed_pairs(
    matches: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand each human counterfactual match into both intervention directions."""
    directed: list[dict[str, Any]] = []
    for match in matches:
        shared = {
            key: value
            for key, value in match.items()
            if key
            not in {
                "positive_example_id",
                "positive_text",
                "positive_prompt",
                "positive_label",
                "positive_review_rating",
                "positive_is_original",
                "negative_example_id",
                "negative_text",
                "negative_prompt",
                "negative_label",
                "negative_review_rating",
                "negative_is_original",
            }
        }
        for target_name, source_name in (("negative", "positive"), ("positive", "negative")):
            directed.append(
                {
                    "case_id": f"{match['pair_id']}-{source_name}-to-{target_name}",
                    **shared,
                    "direction": f"{target_name}_to_{source_name}",
                    "source_example_id": match[f"{source_name}_example_id"],
                    "source_text": match[f"{source_name}_text"],
                    "source_prompt": match[f"{source_name}_prompt"],
                    "source_label": int(source_name == "positive"),
                    "source_label_name": source_name,
                    "source_review_rating": match[f"{source_name}_review_rating"],
                    "source_is_original": match[f"{source_name}_is_original"],
                    "target_example_id": match[f"{target_name}_example_id"],
                    "target_text": match[f"{target_name}_text"],
                    "target_prompt": match[f"{target_name}_prompt"],
                    "target_label": int(target_name == "positive"),
                    "target_label_name": target_name,
                    "target_review_rating": match[f"{target_name}_review_rating"],
                    "target_is_original": match[f"{target_name}_is_original"],
                }
            )
    return directed


def build_counterfactual_configs(
    annotated_rows: Sequence[dict[str, Any]],
    *,
    specs: Sequence[PairingModelSpec],
    pairing_splits: Sequence[str],
    max_prompt_tokens: int,
) -> tuple[dict[str, DatasetDict], dict[str, Any]]:
    pairs = make_original_edit_pairs(annotated_rows)
    flips = [pair for pair in pairs if pair["polarity_flipped"]]
    configs: dict[str, DatasetDict] = {
        "counterfactual_pairs": paired_dataset_dict(pairs),
        "polarity_flip_pairs": paired_dataset_dict(flips),
    }
    counts: dict[str, Any] = {
        "all_original_edit_pairs": len(pairs),
        "polarity_flip_pairs": len(flips),
        "by_split": {
            split: {
                "all_original_edit_pairs": sum(pair["split"] == split for pair in pairs),
                "polarity_flip_pairs": sum(pair["split"] == split for pair in flips),
            }
            for split in ("train", "validation", "test")
        },
        "equal_length": {},
    }
    for name in [*(spec.alias for spec in specs), "common"]:
        matches = make_equal_length_counterfactual_matches(
            annotated_rows,
            specs=specs,
            pairing_model=name,
            splits=pairing_splits,
            max_prompt_tokens=max_prompt_tokens,
        )
        directed = make_counterfactual_directed_pairs(matches)
        slug = "common" if name == "common" else _active_specs(specs, name)[0].column_prefix
        configs[f"{slug}_counterfactual_matched_pairs"] = paired_dataset_dict(matches)
        configs[f"{slug}_counterfactual_directed_pairs"] = paired_dataset_dict(directed)
        counts["equal_length"][name] = {
            "matched_pairs": len(matches),
            "directed_pairs": len(directed),
        }
    return configs, counts


def preprocess_cebab(
    *,
    output_dir: str | Path,
    dataset_name: str = "CEBaB/CEBaB",
    dataset_revision: str | None = None,
    train_split: str = "train_inclusive",
    pairing_models: Sequence[str] | None = None,
    pairing_revisions: Sequence[str] | None = None,
    pairing_splits: Sequence[str] = ("test",),
    max_pairing_prompt_tokens: int = DEFAULT_MAX_PAIRING_PROMPT_TOKENS,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
    private: bool = True,
    hf_token: str | None = None,
    tokenizers: Mapping[str, Any] | None = None,
    source_dataset: DatasetDict | Mapping[str, Any] | None = None,
) -> BinaryPreprocessingResult:
    """Build flat binary, generic pairing, and human-counterfactual CEBaB configs."""
    source = source_dataset
    if source is None:
        source = load_dataset(dataset_name, revision=dataset_revision, token=hf_token)
    rows, source_metadata = cebab_binary_rows(source, train_split=train_split)
    specs = resolve_pairing_models(pairing_models)
    revisions = parse_revision_overrides(pairing_revisions)
    annotated, tokenizer_metadata = annotate_token_lengths(
        rows,
        specs=specs,
        prompt_template=prompt_template,
        revisions=revisions,
        tokenizers=tokenizers,
    )
    generic_configs, generic_counts = build_pairing_configs(
        annotated,
        dataset_name="cebab",
        specs=specs,
        splits=pairing_splits,
        max_prompt_tokens=max_pairing_prompt_tokens,
    )
    counterfactual_configs, counterfactual_counts = build_counterfactual_configs(
        annotated,
        specs=specs,
        pairing_splits=pairing_splits,
        max_prompt_tokens=max_pairing_prompt_tokens,
    )
    datasets = {
        "binary": dataset_dict_by_split(rows),
        **generic_configs,
        **counterfactual_configs,
    }
    metadata = {
        "dataset": dataset_name,
        "requested_dataset_revision": dataset_revision,
        "selected_train_split": train_split,
        "label_policy": {
            "negative": "review_majority 1 or 2",
            "excluded": "review_majority 3 or no majority",
            "positive": "review_majority 4 or 5",
        },
        "label_source": "review_majority (not the EleutherAI median rating)",
        "counterfactual_policy": (
            "pair each retained edited review with the retained original sharing original_id; "
            "polarity_flip_pairs require opposite derived binary labels"
        ),
        "correctness_filter": None,
        "prompt_template": prompt_template,
        "pairing_splits": list(pairing_splits),
        "max_pairing_prompt_tokens": max_pairing_prompt_tokens,
        "pairing_models": [spec.alias for spec in specs],
        "tokenizers": tokenizer_metadata,
        "counts": {
            **source_metadata,
            "binary_rows": len(rows),
            "binary_by_split": {
                split: {
                    "rows": sum(row["split"] == split for row in rows),
                    "negative": sum(
                        row["split"] == split and row["label"] == 0 for row in rows
                    ),
                    "positive": sum(
                        row["split"] == split and row["label"] == 1 for row in rows
                    ),
                }
                for split in ("train", "validation", "test")
            },
            "pairing": generic_counts,
            "counterfactual": counterfactual_counts,
        },
    }
    return finish_preprocessing(
        dataset_name="CEBaB Binary and Human Counterfactuals",
        output_dir=output_dir,
        datasets=datasets,
        metadata=metadata,
        push_to_hub=push_to_hub,
        hub_repo_id=hub_repo_id,
        default_repo_name="sentiment-manifold-cebab-binary",
        private=private,
        hf_token=hf_token,
    )
