"""Paper-grounded SST preprocessing with Pythia-1.4B.

This module adapts ``eliciting-latent-sentiment/utils/treebank.py`` while making
the intermediate datasets explicit and correcting the clean/corrupt pairing to
preserve opposite labels and equal token lengths for every directed example.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...devices import resolve_device
from .common import (
    annotate_token_lengths,
    build_pairing_configs,
    parse_revision_overrides,
    resolve_pairing_models,
)


PYTHIA_MODEL = "EleutherAI/pythia-1.4b"
POSITIVE_ANSWER = " Positive"
NEGATIVE_ANSWER = " Negative"
NEUTRAL_LOWER = 0.4
NEUTRAL_UPPER = 0.6
SCAFFOLD_TEMPLATE = "Review Text: {text} Review Sentiment:"
TIGGES_BINARIZATION = "tigges"
NEUTRAL_REMOVED_BINARIZATION = "neutral_removed"


@dataclass(frozen=True)
class SSTPreprocessingResult:
    output_dir: Path
    datasets: dict[str, DatasetDict]
    metadata: dict[str, Any]
    hub_repo_id: str | None = None


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _five_class_label(score: float) -> str:
    if score <= 0.2:
        return "very_negative"
    if score <= 0.4:
        return "negative"
    if score <= 0.6:
        return "neutral"
    if score <= 0.8:
        return "positive"
    return "very_positive"


def binary_label_without_neutral(score: float) -> int | None:
    """Collapse SST scores after removing the official neutral interval.

    SST defines ``(0.4, 0.6]`` as neutral. Score ``0.4`` remains negative,
    while positive scores are strictly greater than ``0.6``.
    """
    if score <= NEUTRAL_LOWER:
        return 0
    if score <= NEUTRAL_UPPER:
        return None
    return 1


def tigges_binary_label(score: float) -> int:
    """Match upstream ``int(round(score))`` including its 0.5 -> 0 tie."""
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"SST sentiment score must be in [0, 1], got {score}")
    return int(round(score))


def binarize_rows(
    rows: Iterable[dict[str, Any]], method: str
) -> list[dict[str, Any]]:
    """Apply one declared SST binary-label policy to source sentence rows."""
    if method not in {TIGGES_BINARIZATION, NEUTRAL_REMOVED_BINARIZATION}:
        raise ValueError(
            f"Unknown SST binarization {method!r}; expected 'tigges' or 'neutral_removed'"
        )
    result: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        if method == TIGGES_BINARIZATION:
            label = tigges_binary_label(float(row["sentiment_score"]))
        else:
            label = binary_label_without_neutral(float(row["sentiment_score"]))
            if label is None:
                continue
        row.update(
            label=int(label),
            label_name="positive" if label else "negative",
            binarization_method=method,
            is_neutral_removed=False,
        )
        result.append(row)
    return result


def load_sst_source_sentences(root: str | Path) -> list[dict[str, Any]]:
    """Join SST source-sentence rows with phrase scores and official splits."""
    root = Path(root)
    dictionary_path = root / "dictionary_fixed.txt"
    if not dictionary_path.exists():
        dictionary_path = root / "dictionary.txt"
    sentences_path = root / "datasetSentences_fixed.txt"
    if not sentences_path.exists():
        sentences_path = root / "datasetSentences.txt"

    required = [
        dictionary_path,
        root / "sentiment_labels.txt",
        sentences_path,
        root / "datasetSplit.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stanford Sentiment Treebank files: {missing}")

    scores_by_id: dict[int, float] = {}
    with (root / "sentiment_labels.txt").open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            phrase_id, score = line.rstrip("\n").split("|")
            scores_by_id[int(phrase_id)] = float(score)

    phrase_rows: dict[str, tuple[int, float]] = {}
    with dictionary_path.open(encoding="utf-8") as handle:
        for line in handle:
            phrase, phrase_id_text = line.rstrip("\n").rsplit("|", 1)
            phrase_id = int(phrase_id_text)
            if phrase not in phrase_rows and phrase_id in scores_by_id:
                phrase_rows[phrase] = (phrase_id, scores_by_id[phrase_id])

    split_by_id: dict[int, str] = {}
    split_names = {1: "train", 2: "test", 3: "validation"}
    with (root / "datasetSplit.txt").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split_by_id[int(row["sentence_index"])] = split_names[int(row["splitset_label"])]

    rows: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    with sentences_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sentence_index = int(row["sentence_index"])
            text = row["sentence"]
            if text in seen_texts or text not in phrase_rows or sentence_index not in split_by_id:
                continue
            seen_texts.add(text)
            phrase_id, score = phrase_rows[text]
            label = binary_label_without_neutral(score)
            rows.append(
                {
                    "example_id": f"sst-{sentence_index}",
                    "sentence_index": sentence_index,
                    "phrase_id": phrase_id,
                    "text": text,
                    "sentiment_score": score,
                    "original_label_5": _five_class_label(score),
                    "label": label,
                    "label_name": None if label is None else ("positive" if label else "negative"),
                    "split": split_by_id[sentence_index],
                    "is_neutral_removed": label is None,
                }
            )
    return rows


def remove_neutral(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return source sentences outside SST's ``(0.4, 0.6]`` neutral band."""
    return [dict(row) for row in rows if row["label"] is not None]


def apply_labels_to_scored_rows(
    scored_rows: Iterable[dict[str, Any]], labeled_rows: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach a label policy to shared Pythia scores and recompute correctness."""
    labels = {row["example_id"]: row for row in labeled_rows}
    result: list[dict[str, Any]] = []
    for scored_row in scored_rows:
        labeled = labels.get(scored_row["example_id"])
        if labeled is None:
            continue
        row = {**scored_row, **labeled}
        logit_diff = float(row["positive_minus_negative_logit"])
        row["signed_correct_logit_diff"] = logit_diff if row["label"] == 1 else -logit_diff
        row["pythia_correct"] = int(row["predicted_label"]) == int(row["label"])
        result.append(row)
    return result


def _single_token_id(tokenizer, text: str) -> int:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(f"Expected {text!r} to be one Pythia token, got {token_ids}")
    return int(token_ids[0])


def _token_ids(tokenizer, text: str) -> list[int]:
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = tokenizer.eos_token_id
    if bos_token_id is None:
        raise ValueError("Pythia tokenizer has neither a BOS nor EOS token")
    return [int(bos_token_id), *map(int, ids)]


def score_test_sentences_with_pythia(
    rows: Iterable[dict[str, Any]],
    *,
    device: str = "auto",
    dtype: str = "auto",
    batch_size: int = 16,
    revision: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score labeled SST test sentences with the paper's Pythia classifier."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    test_rows = [dict(row) for row in rows if row["split"] == "test"]
    device_spec = resolve_device(device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(PYTHIA_MODEL, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        PYTHIA_MODEL,
        revision=revision,
        dtype=device_spec.dtype,
    ).to(device_spec.device)
    model.eval()
    model.requires_grad_(False)

    positive_id = _single_token_id(tokenizer, POSITIVE_ANSWER)
    negative_id = _single_token_id(tokenizer, NEGATIVE_ANSWER)
    scored: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(test_rows), batch_size), desc="Pythia SST filtering"):
        batch_rows = test_rows[start : start + batch_size]
        prompts = [SCAFFOLD_TEMPLATE.format(text=row["text"]) for row in batch_rows]
        encoded_rows = [_token_ids(tokenizer, prompt) for prompt in prompts]
        encoded = tokenizer.pad(
            {"input_ids": encoded_rows},
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device_spec.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded, use_cache=False).logits[:, -1, :].float().cpu()
        positive_logits = logits[:, positive_id]
        negative_logits = logits[:, negative_id]
        for index, (row, prompt, prompt_ids) in enumerate(zip(batch_rows, prompts, encoded_rows)):
            positive_logit = float(positive_logits[index])
            negative_logit = float(negative_logits[index])
            positive_minus_negative = positive_logit - negative_logit
            predicted_label = int(positive_minus_negative > 0)
            raw_ids = _token_ids(tokenizer, row["text"])
            scored.append(
                {
                    **row,
                    "prompt": prompt,
                    "pythia_raw_num_tokens": len(raw_ids),
                    "pythia_prompt_num_tokens": len(prompt_ids),
                    "positive_token_id": positive_id,
                    "negative_token_id": negative_id,
                    "positive_logit": positive_logit,
                    "negative_logit": negative_logit,
                    "positive_minus_negative_logit": positive_minus_negative,
                    "signed_correct_logit_diff": (
                        positive_minus_negative if row["label"] == 1 else -positive_minus_negative
                    ),
                    "predicted_label": predicted_label,
                    "predicted_label_name": "positive" if predicted_label else "negative",
                    "pythia_correct": predicted_label == row["label"],
                }
            )

    metadata = {
        "model_name": PYTHIA_MODEL,
        "requested_revision": revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "device": str(device_spec.device),
        "dtype": str(device_spec.dtype),
        "prepend_bos": True,
        "positive_answer": POSITIVE_ANSWER,
        "negative_answer": NEGATIVE_ANSWER,
        "positive_token_id": positive_id,
        "negative_token_id": negative_id,
        "scaffold_template": SCAFFOLD_TEMPLATE,
    }
    del model
    return scored, metadata


def make_maximal_matches(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make maximal deterministic, non-reused opposite-label matches."""
    rows = [dict(row) for row in rows]
    methods = {row.get("binarization_method", "unspecified") for row in rows}
    if len(methods) > 1:
        raise ValueError("Pairing rows must use one SST binarization method")
    method = next(iter(methods), "unspecified")
    buckets: dict[tuple[int, int], dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        if not row["pythia_correct"]:
            continue
        key = (row["pythia_raw_num_tokens"], row["pythia_prompt_num_tokens"])
        buckets.setdefault(key, {0: [], 1: []})[int(row["label"])].append(dict(row))

    matches: list[dict[str, Any]] = []
    for raw_length, prompt_length in sorted(buckets):
        negative = sorted(
            buckets[(raw_length, prompt_length)][0], key=lambda row: row["sentence_index"]
        )
        positive = sorted(
            buckets[(raw_length, prompt_length)][1], key=lambda row: row["sentence_index"]
        )
        for pos, neg in zip(positive, negative):
            pair_id = f"sst-{method}-pythia-pair-{len(matches):05d}"
            matches.append(
                {
                    "pair_id": pair_id,
                    "binarization_method": method,
                    "split": "test",
                    "pythia_raw_num_tokens": raw_length,
                    "pythia_prompt_num_tokens": prompt_length,
                    "positive_example_id": pos["example_id"],
                    "positive_sentence_index": pos["sentence_index"],
                    "positive_phrase_id": pos["phrase_id"],
                    "positive_text": pos["text"],
                    "positive_prompt": pos["prompt"],
                    "positive_sentiment_score": pos["sentiment_score"],
                    "positive_signed_correct_logit_diff": pos["signed_correct_logit_diff"],
                    "negative_example_id": neg["example_id"],
                    "negative_sentence_index": neg["sentence_index"],
                    "negative_phrase_id": neg["phrase_id"],
                    "negative_text": neg["text"],
                    "negative_prompt": neg["prompt"],
                    "negative_sentiment_score": neg["sentiment_score"],
                    "negative_signed_correct_logit_diff": neg["signed_correct_logit_diff"],
                }
            )
    return matches


def make_directed_pairs(matches: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Represent each match in both clean/corrupted patching directions."""
    directed: list[dict[str, Any]] = []
    for match in matches:
        for clean_polarity, corrupted_polarity in (
            ("positive", "negative"),
            ("negative", "positive"),
        ):
            clean_label = int(clean_polarity == "positive")
            directed.append(
                {
                    "case_id": (
                        f"{match['pair_id']}-{corrupted_polarity}-to-{clean_polarity}"
                    ),
                    "pair_id": match["pair_id"],
                    "split": match["split"],
                    "direction": f"{corrupted_polarity}_to_{clean_polarity}",
                    "pythia_raw_num_tokens": match["pythia_raw_num_tokens"],
                    "pythia_prompt_num_tokens": match["pythia_prompt_num_tokens"],
                    "clean_example_id": match[f"{clean_polarity}_example_id"],
                    "clean_text": match[f"{clean_polarity}_text"],
                    "clean_prompt": match[f"{clean_polarity}_prompt"],
                    "clean_label": clean_label,
                    "clean_label_name": clean_polarity,
                    "clean_answer": POSITIVE_ANSWER.strip()
                    if clean_label
                    else NEGATIVE_ANSWER.strip(),
                    "corrupted_example_id": match[f"{corrupted_polarity}_example_id"],
                    "corrupted_text": match[f"{corrupted_polarity}_text"],
                    "corrupted_prompt": match[f"{corrupted_polarity}_prompt"],
                    "corrupted_label": 1 - clean_label,
                    "corrupted_label_name": corrupted_polarity,
                    "corrupted_answer": NEGATIVE_ANSWER.strip()
                    if clean_label
                    else POSITIVE_ANSWER.strip(),
                }
            )
    return directed


def _dataset_dict_by_split(rows: list[dict[str, Any]]) -> DatasetDict:
    return DatasetDict(
        {
            split: Dataset.from_list([row for row in rows if row["split"] == split])
            for split in ("train", "validation", "test")
            if any(row["split"] == split for row in rows)
        }
    )


def _single_test_dataset(rows: list[dict[str, Any]]) -> DatasetDict:
    return DatasetDict({"test": Dataset.from_list(rows)})


def _dataset_card(metadata: dict[str, Any], counts: dict[str, Any]) -> str:
    config_lines = "\n".join(
        f"- config_name: {name}" for name in metadata["dataset_configs"]
    )
    return f"""---
language:
- en
task_categories:
- text-classification
pretty_name: Sentiment Manifold SST Pythia-1.4B Preprocessing
license: other
configs:
{config_lines}
---

# Sentiment Manifold SST Pythia-1.4B Preprocessing

Source-sentence preprocessing for reproducing the SST directional-patching setup in Tigges et al.,
*Language Models Linearly Represent Sentiment*. The source is Stanford Sentiment Treebank v1.0.
Consult the original SST distribution for its terms and citation requirements.

## Binary-label variants

`tigges` follows the accompanying code's `int(round(score))` behavior: scores `<= 0.5` are
negative and scores `> 0.5` are positive. `neutral_removed` preserves the alternative policy:
scores `<= 0.4` are negative, `(0.4, 0.6]` is removed, and scores `> 0.6` are positive.

## Configurations

Each variant has `*_binarized`, `*_pythia_scored`, `*_pythia_correct`, `*_matched_pairs`,
and `*_directed_pairs` configurations. When both variants are requested, Pythia is scored once on
all Tigges-binarized test rows and correctness is computed independently for each label policy.

No GPT-4 labels or manual pair validation are used. Matches are opposite according to SST's human
labels; they are not minimal semantic rewrites. The legacy pair configurations preserve the
paper-oriented Pythia pairing. RQ2 additionally writes `*_pairing_candidates`, tokenizer-specific
pairs for GPT-2, Qwen, Gemma, and Pythia when selected, and `*_common_*` pairs whose full prompts
have equal length under every selected tokenizer.

## Counts

```json
{json.dumps(counts, indent=2, sort_keys=True)}
```

## Model and preprocessing metadata

```json
{json.dumps(metadata, indent=2, sort_keys=True)}
```

## Citation

Please cite both Tigges et al. (2024) and Socher et al. (2013) when using this dataset.
"""


def publish_dataset_configs(
    datasets: dict[str, DatasetDict],
    *,
    repo_id: str | None,
    private: bool,
    card_text: str,
    token: str,
) -> str:
    """Create a Hub dataset repository and upload every intermediate config."""
    if not token:
        raise ValueError("A Hugging Face token is required to publish the SST datasets")
    api = HfApi(token=token)
    if repo_id is None:
        account = api.whoami()["name"]
        repo_id = f"{account}/sentiment-manifold-sst-pythia-1.4b"
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
        commit_message="Document SST preprocessing and dataset configurations",
    )
    return repo_id


def preprocess_sst(
    *,
    sst_root: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    dtype: str = "auto",
    batch_size: int = 16,
    revision: str | None = None,
    binarization: str = "both",
    pairing_models: Sequence[str] | None = None,
    pairing_revisions: Sequence[str] | None = None,
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
    private: bool = True,
    hf_token: str | None = None,
    tokenizers: Mapping[str, Any] | None = None,
) -> SSTPreprocessingResult:
    """Run, save, and optionally publish the complete SST preprocessing pipeline."""
    if push_to_hub:
        if not hf_token:
            raise ValueError(
                "Publishing requires a Hugging Face token supplied through the configured "
                "environment variable or HF_TOKEN_PATH"
            )
        # Fail before downloading and scoring with Pythia when credentials are invalid.
        HfApi(token=hf_token).whoami()

    sst_root = Path(sst_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_methods = {
        "both": (TIGGES_BINARIZATION, NEUTRAL_REMOVED_BINARIZATION),
        TIGGES_BINARIZATION: (TIGGES_BINARIZATION,),
        NEUTRAL_REMOVED_BINARIZATION: (NEUTRAL_REMOVED_BINARIZATION,),
    }
    if binarization not in requested_methods:
        raise ValueError("binarization must be 'both', 'tigges', or 'neutral_removed'")

    source_rows = load_sst_source_sentences(sst_root)
    variants = {
        method: binarize_rows(source_rows, method)
        for method in requested_methods[binarization]
    }
    scoring_method = (
        TIGGES_BINARIZATION
        if TIGGES_BINARIZATION in variants
        else NEUTRAL_REMOVED_BINARIZATION
    )
    shared_scored, model_metadata = score_test_sentences_with_pythia(
        variants[scoring_method],
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        revision=revision,
    )
    pairing_specs = resolve_pairing_models(pairing_models)
    parsed_pairing_revisions = parse_revision_overrides(pairing_revisions)

    datasets: dict[str, DatasetDict] = {}
    variant_counts: dict[str, Any] = {}
    pairing_tokenizer_metadata: dict[str, Any] = {}
    for method, binarized in variants.items():
        scored = apply_labels_to_scored_rows(shared_scored, binarized)
        correct = [row for row in scored if row["pythia_correct"]]
        if any(not row["pythia_correct"] for row in correct):
            raise AssertionError("Pythia-correct SST configuration contains an incorrect row")
        matches = make_maximal_matches(correct)
        directed = make_directed_pairs(matches)
        pairing_candidates, pairing_tokenizer_metadata = annotate_token_lengths(
            correct,
            specs=pairing_specs,
            prompt_template=SCAFFOLD_TEMPLATE,
            revisions=parsed_pairing_revisions,
            tokenizers=tokenizers,
        )
        pairing_configs, pairing_counts = build_pairing_configs(
            pairing_candidates,
            dataset_name=f"sst-{method}",
            specs=pairing_specs,
            splits=("test",),
        )
        datasets.update(
            {
                f"{method}_binarized": _dataset_dict_by_split(binarized),
                f"{method}_pythia_scored": _single_test_dataset(scored),
                f"{method}_pythia_correct": _single_test_dataset(correct),
                f"{method}_matched_pairs": _single_test_dataset(matches),
                f"{method}_directed_pairs": _single_test_dataset(directed),
                **{f"{method}_{name}": value for name, value in pairing_configs.items()},
            }
        )
        variant_counts[method] = {
            "binarized": {
                split: sum(row["split"] == split for row in binarized)
                for split in ("train", "validation", "test")
            },
            "pythia_scored_test": len(scored),
            "pythia_correct_test": len(correct),
            "pythia_incorrect_test": len(scored) - len(correct),
            "matched_pairs": len(matches),
            "directed_pairs": len(directed),
            "tokenizer_specific_pairing": pairing_counts,
        }
    for config_name, dataset_dict in datasets.items():
        dataset_dict.save_to_disk(output_dir / config_name)

    source_files = {
        name: _sha256(sst_root / name)
        for name in (
            "sentiment_labels.txt",
            "datasetSplit.txt",
            "dictionary_fixed.txt",
            "dictionary.txt",
            "datasetSentences_fixed.txt",
            "datasetSentences.txt",
        )
        if (sst_root / name).exists()
    }
    counts = {"source_sentences": len(source_rows), "variants": variant_counts}
    metadata = {
        **model_metadata,
        "sst_root": str(sst_root),
        "source_files_sha256": source_files,
        "source_rows": (
            "datasetSentences_fixed.txt joined to dictionary_fixed.txt and sentiment_labels.txt"
        ),
        "split_mapping": {"1": "train", "2": "test", "3": "validation"},
        "binary_collapse": {
            TIGGES_BINARIZATION: {
                "negative": "score <= 0.5",
                "positive": "score > 0.5",
                "upstream_equivalent": "int(round(score))",
            },
            NEUTRAL_REMOVED_BINARIZATION: {
                "negative": "score <= 0.4",
                "neutral_removed": "0.4 < score <= 0.6",
                "positive": "score > 0.6",
            },
        },
        "dataset_configs": list(datasets),
        "requested_binarization": binarization,
        "pairing_models": [spec.alias for spec in pairing_specs],
        "pairing_tokenizers": pairing_tokenizer_metadata,
        "pairing": (
            "maximal deterministic non-reused opposite-label matches by equal raw and prompt "
            "Pythia token counts"
        ),
        "counts": counts,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    card_text = _dataset_card(metadata, counts)
    (output_dir / "README.md").write_text(card_text)

    published_repo = None
    if push_to_hub:
        assert hf_token is not None
        published_repo = publish_dataset_configs(
            datasets,
            repo_id=hub_repo_id,
            private=private,
            card_text=card_text,
            token=hf_token,
        )
        metadata["hub_repo_id"] = published_repo
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    return SSTPreprocessingResult(output_dir, datasets, metadata, published_repo)
