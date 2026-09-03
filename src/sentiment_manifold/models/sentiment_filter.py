"""Zero-shot binary-sentiment scoring used by preprocessing pipelines."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..devices import resolve_device


PYTHIA_FILTER_MODELS = {
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
}
DEFAULT_PYTHIA_FILTER_MODEL = "pythia-2.8b"
TIGGES_PYTHIA_FILTER_MODEL = "pythia-1.4b"
POSITIVE_ANSWER = " Positive"
NEGATIVE_ANSWER = " Negative"
DEFAULT_SENTIMENT_PROMPT = "Review Text: {text} Review Sentiment:"


def resolve_pythia_filter_model(model_name: str) -> tuple[str, str]:
    """Return a stable alias and Hugging Face model name for a supported Pythia filter."""
    if model_name in PYTHIA_FILTER_MODELS:
        return model_name, PYTHIA_FILTER_MODELS[model_name]
    for alias, hub_name in PYTHIA_FILTER_MODELS.items():
        if model_name == hub_name:
            return alias, hub_name
    raise ValueError(
        f"Unsupported correctness-filter model {model_name!r}; "
        f"choose from {list(PYTHIA_FILTER_MODELS)}"
    )


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


def score_binary_rows_with_pythia(
    rows: Iterable[dict[str, Any]],
    *,
    splits: Sequence[str] = ("test",),
    prompt_template: str = DEFAULT_SENTIMENT_PROMPT,
    model_name: str = DEFAULT_PYTHIA_FILTER_MODEL,
    device: str = "auto",
    dtype: str = "auto",
    batch_size: int = 16,
    revision: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score gold-labelled rows without changing their labels.

    The returned ``pythia_correct`` flag records whether the larger-probability
    ``Positive``/``Negative`` answer agrees with the dataset label. Callers may
    use that flag for selection, but the prediction never replaces ground truth.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if "{text}" not in prompt_template:
        raise ValueError("prompt_template must contain the literal placeholder '{text}'")
    alias, hub_name = resolve_pythia_filter_model(model_name)
    allowed_splits = set(splits)
    selected_rows = [dict(row) for row in rows if str(row["split"]) in allowed_splits]
    for row in selected_rows:
        if int(row["label"]) not in (0, 1):
            raise ValueError(f"Correctness filtering requires labels 0/1, got {row['label']}")

    device_spec = resolve_device(device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(hub_name, revision=revision, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        hub_name,
        revision=revision,
        dtype=device_spec.dtype,
    ).to(device_spec.device)
    model.eval()
    model.requires_grad_(False)

    configured_context = getattr(model.config, "max_position_embeddings", None)
    tokenizer_context = getattr(tokenizer, "model_max_length", None)
    context_value = configured_context or tokenizer_context
    if context_value is None or int(context_value) <= 0:
        raise ValueError(f"Could not determine a positive context length for {hub_name}")
    context_length = int(context_value)

    positive_id = _single_token_id(tokenizer, POSITIVE_ANSWER)
    negative_id = _single_token_id(tokenizer, NEGATIVE_ANSWER)
    prepared: list[tuple[dict[str, Any], str, list[int], int]] = []
    over_context_rows = 0
    for row in selected_rows:
        prompt = prompt_template.format(text=row["text"])
        prompt_ids = _token_ids(tokenizer, prompt)
        if len(prompt_ids) > context_length:
            over_context_rows += 1
            continue
        raw_length = len(_token_ids(tokenizer, str(row["text"])))
        prepared.append((row, prompt, prompt_ids, raw_length))

    scored: list[dict[str, Any]] = []
    description = f"{alias} correctness filtering"
    for start in tqdm(range(0, len(prepared), batch_size), desc=description):
        batch = prepared[start : start + batch_size]
        encoded_rows = [prompt_ids for _, _, prompt_ids, _ in batch]
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
        for index, (row, prompt, prompt_ids, raw_length) in enumerate(batch):
            positive_logit = float(positive_logits[index])
            negative_logit = float(negative_logits[index])
            positive_minus_negative = positive_logit - negative_logit
            predicted_label = int(positive_minus_negative > 0)
            scored.append(
                {
                    **row,
                    "prompt": prompt,
                    "filter_model_alias": alias,
                    "filter_model_name": hub_name,
                    "pythia_raw_num_tokens": raw_length,
                    "pythia_prompt_num_tokens": len(prompt_ids),
                    "positive_token_id": positive_id,
                    "negative_token_id": negative_id,
                    "positive_logit": positive_logit,
                    "negative_logit": negative_logit,
                    "positive_minus_negative_logit": positive_minus_negative,
                    "signed_correct_logit_diff": (
                        positive_minus_negative
                        if int(row["label"]) == 1
                        else -positive_minus_negative
                    ),
                    "predicted_label": predicted_label,
                    "predicted_label_name": "positive" if predicted_label else "negative",
                    "pythia_correct": predicted_label == int(row["label"]),
                }
            )

    metadata = {
        # Compatibility keys retained for SST artifacts created by the original preprocessor.
        "model_name": hub_name,
        "requested_revision": revision,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "resolved_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "filter_model_alias": alias,
        "filter_model_name": hub_name,
        "requested_filter_revision": revision,
        "resolved_filter_model_revision": getattr(model.config, "_commit_hash", None),
        "resolved_filter_tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "filter_splits": list(splits),
        "filter_context_length": context_length,
        "filter_input_rows": len(selected_rows),
        "filter_scored_rows": len(scored),
        "filter_over_context_rows": over_context_rows,
        "filter_device": str(device_spec.device),
        "filter_dtype": str(device_spec.dtype),
        "filter_prepend_bos": True,
        "positive_answer": POSITIVE_ANSWER,
        "negative_answer": NEGATIVE_ANSWER,
        "positive_token_id": positive_id,
        "negative_token_id": negative_id,
        "scaffold_template": prompt_template,
    }
    del model
    return scored, metadata
