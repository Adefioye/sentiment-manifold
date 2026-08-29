"""End-to-end all-layer reproduction orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .artifacts import DirectionArtifact, artifact_path
from .config import ReproductionConfig
from .data import load_openwebtext, load_sst, load_toy_movie_review, pair_sst_by_token_length
from .devices import clear_device_cache, resolve_device
from .directions import create_fitter
from .directions.das import DASFitter, DASTrainingConfig
from .evaluation import (
    cosine_similarity_table,
    evaluate_directional_patching,
    evaluate_openwebtext_ablation,
    projection_accuracy,
)
from .evaluation.projections import projection_threshold
from .models import CausalLMAdapter


SST_CONTINUATION_ANSWERS = {1: (" good",), 0: (" bad",)}
ARTIFACT_SCHEMA_VERSION = 3


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")


def _write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _filter_single_focus(adapter: CausalLMAdapter, examples):
    return [example for example in examples if adapter.focus_is_single_token(example)]


def _prompt_manifest_rows(adapter: CausalLMAdapter, toy, retained_ids: set[str]) -> list[dict]:
    rows: list[dict] = []
    for split, examples in (("train", toy.train), ("test", toy.test)):
        for example in examples:
            tokenized = adapter.tokenize([example])
            mask = tokenized.attention_mask[0].bool()
            token_ids = tokenized.input_ids[0][mask].tolist()
            rows.append(
                {
                    "example_id": example.example_id,
                    "split": split,
                    "label": example.label,
                    "retained": example.example_id in retained_ids,
                    "prompt_text": example.text,
                    "prompt_repr": repr(example.text),
                    "prompt_utf8_hex": example.text.encode("utf-8").hex(),
                    "prompt_sha256": hashlib.sha256(example.text.encode("utf-8")).hexdigest(),
                    "token_ids": json.dumps(token_ids),
                    "num_tokens": len(token_ids),
                    "focus_position": None
                    if tokenized.focus_positions is None
                    else int(tokenized.focus_positions[0]),
                    "adjective": example.metadata.get("adjective"),
                    "verb": example.metadata.get("verb"),
                }
            )
    return rows


def _answer_manifest_rows(adapter: CausalLMAdapter, toy_answers) -> list[dict]:
    rows: list[dict] = []
    for dataset, answers in (
        ("toy_movie_review", toy_answers),
        ("sst_continuation", SST_CONTINUATION_ANSWERS),
    ):
        for label, values in answers.items():
            for pair_index, answer in enumerate(values):
                rows.append(
                    {
                        "dataset": dataset,
                        "label": label,
                        "pair_index": pair_index,
                        "answer": answer,
                        "answer_repr": repr(answer),
                        "token_id": adapter.single_token_id(answer),
                    }
                )
    return rows


def _vocabulary_manifest_rows(adapter: CausalLMAdapter, raw_toy, filtered_toy) -> list[dict]:
    rows: list[dict] = []
    for split in ("train", "test"):
        for label in (1, 0):
            retained = set(filtered_toy.adjectives[split][label])
            for word in raw_toy.adjectives[split][label]:
                token_ids = adapter.tokenizer(" " + word.strip(), add_special_tokens=False)[
                    "input_ids"
                ]
                rows.append(
                    {
                        "split": split,
                        "label": label,
                        "word_type": "adjective",
                        "word": word,
                        "retained": word in retained,
                        "token_ids": json.dumps(token_ids),
                        "num_tokens": len(token_ids),
                    }
                )
    for label in (1, 0):
        retained = set(filtered_toy.verbs[label])
        for word in raw_toy.verbs[label]:
            token_ids = adapter.tokenizer(" " + word.strip(), add_special_tokens=False)["input_ids"]
            rows.append(
                {
                    "split": "all",
                    "label": label,
                    "word_type": "verb",
                    "word": word,
                    "retained": word in retained,
                    "token_ids": json.dumps(token_ids),
                    "num_tokens": len(token_ids),
                }
            )
    return rows


def _pair_manifest_rows(adapter: CausalLMAdapter, dataset: str, pairs) -> list[dict]:
    rows: list[dict] = []
    for pair_index, pair in enumerate(pairs):
        clean = adapter.tokenize([pair.clean])
        corrupted = adapter.tokenize([pair.corrupted])
        clean_ids = clean.input_ids[0][clean.attention_mask[0].bool()].tolist()
        corrupted_ids = corrupted.input_ids[0][corrupted.attention_mask[0].bool()].tolist()
        rows.append(
            {
                "dataset": dataset,
                "pair_index": pair_index,
                "clean_id": pair.clean.example_id,
                "corrupted_id": pair.corrupted.example_id,
                "clean_label": pair.clean.label,
                "corrupted_label": pair.corrupted.label,
                "clean_text": pair.clean.text,
                "corrupted_text": pair.corrupted.text,
                "clean_sha256": hashlib.sha256(pair.clean.text.encode("utf-8")).hexdigest(),
                "corrupted_sha256": hashlib.sha256(pair.corrupted.text.encode("utf-8")).hexdigest(),
                "clean_token_ids": json.dumps(clean_ids),
                "corrupted_token_ids": json.dumps(corrupted_ids),
                "clean_num_tokens": len(clean_ids),
                "corrupted_num_tokens": len(corrupted_ids),
                "equal_token_length": len(clean_ids) == len(corrupted_ids),
            }
        )
    return rows


def _artifact_is_compatible(artifact: DirectionArtifact, method: str) -> bool:
    if artifact.metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        return False
    if method.startswith("das"):
        return artifact.metadata.get("implementation") == "tigges_rotation"
    return True


def run_reproduction(config: ReproductionConfig) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device_spec = resolve_device(config.model.device, config.model.dtype)
    adapter = CausalLMAdapter.from_pretrained(
        config.model.hub_name,
        device_spec,
        revision=config.model.revision,
        prepend_bos=config.model.prepend_bos,
    )
    run_dir = Path(config.experiment.output_dir) / _slug(config.model.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = config.to_dict()
    resolved["runtime"] = adapter.provenance()
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, sort_keys=True))

    raw_toy = load_toy_movie_review(config.data.toy_config)
    toy = raw_toy.tokenizer_filtered(adapter.tokenizer)
    toy_train = _filter_single_focus(adapter, toy.train)
    toy_test = _filter_single_focus(adapter, toy.test)
    allowed_train = {example.example_id for example in toy_train}
    allowed_test = {example.example_id for example in toy_test}
    train_pairs = [
        pair
        for pair in toy.paired("train")
        if pair.clean.example_id in allowed_train and pair.corrupted.example_id in allowed_train
    ]
    test_pairs = [
        pair
        for pair in toy.paired("test")
        if pair.clean.example_id in allowed_test and pair.corrupted.example_id in allowed_test
    ]
    if not toy_train or not toy_test:
        raise RuntimeError("Tokenizer filtering removed every ToyMovieReview example")
    retained_ids = {example.example_id for example in (*toy_train, *toy_test)}
    _write_rows(_prompt_manifest_rows(adapter, toy, retained_ids), run_dir / "prompt_manifest.csv")
    _write_rows(_answer_manifest_rows(adapter, toy.answers), run_dir / "answer_tokens.csv")
    _write_rows(_vocabulary_manifest_rows(adapter, raw_toy, toy), run_dir / "toy_vocabulary.csv")

    sst_examples = load_sst(config.data.sst_root, config.data.sst_split)
    if config.data.sst_max_examples is not None:
        sst_examples_for_projection = sst_examples[: config.data.sst_max_examples]
    else:
        sst_examples_for_projection = sst_examples
    sst_pairs = pair_sst_by_token_length(
        sst_examples,
        adapter.tokenizer,
        max_pairs=config.data.sst_max_pairs,
        seed=config.seed,
    )
    if not sst_pairs:
        raise RuntimeError("No equal-token-length SST counterfactual pairs were constructed")
    pair_rows = [
        *_pair_manifest_rows(adapter, "toy_train", train_pairs),
        *_pair_manifest_rows(adapter, "toy_test", test_pairs),
        *_pair_manifest_rows(adapter, f"sst_{config.data.sst_split}", sst_pairs),
    ]
    _write_rows(pair_rows, run_dir / "pair_manifest.csv")

    owt_texts = None
    if config.experiment.resample_ablation_enabled:
        owt_texts = load_openwebtext(
            config.data.openwebtext_dataset,
            config.data.openwebtext_split,
            config.data.openwebtext_max_samples,
        )

    metrics_rows: list[dict] = []
    similarity_rows: list[dict] = []
    patching_rows: list[dict] = []
    owt_control_rows: list[dict] = []
    das_loss_rows: list[dict] = []
    direction_metadata_rows: list[dict] = []
    layers = config.layers_for(adapter.n_layers)
    labels_train = np.asarray([example.label for example in toy_train])
    labels_test = np.asarray([example.label for example in toy_test])
    labels_sst = np.asarray([example.label for example in sst_examples_for_projection])

    for layer in tqdm(layers, desc=f"{config.model.name} layer boundaries"):
        train_activations = adapter.extract_activations(
            toy_train, layer, position="focus", batch_size=config.model.batch_size
        )
        test_activations = adapter.extract_activations(
            toy_test, layer, position="focus", batch_size=config.model.batch_size
        )
        sst_activations = adapter.extract_activations(
            sst_examples_for_projection,
            layer,
            position="final",
            batch_size=config.model.batch_size,
        )
        layer_directions: dict[str, np.ndarray] = {}
        for method in config.experiment.methods:
            path = artifact_path(run_dir, method, layer)
            if config.experiment.resume and path.exists():
                artifact = DirectionArtifact.load(path)
                if not _artifact_is_compatible(artifact, method):
                    artifact = None
            else:
                artifact = None
            if artifact is None:
                if method in {"das", "das2d", "das3d"}:
                    if config.das.implementation != "tigges_rotation":
                        raise ValueError(
                            "Replication methods das/das2d/das3d require "
                            "das.implementation=tigges_rotation"
                        )
                    dimension = {"das": 1, "das2d": 2, "das3d": 3}[method]
                    fitter = DASFitter(
                        DASTrainingConfig(
                            epochs=config.das.epochs,
                            learning_rate=config.das.learning_rate,
                            weight_decay=config.das.weight_decay,
                            batch_size=config.das.batch_size,
                            max_grad_norm=config.das.max_grad_norm,
                            seed=config.seed,
                        ),
                        dimension=dimension,
                        name=method,
                    )
                    result = fitter.fit(adapter, train_pairs, layer=layer, answers=toy.answers)
                else:
                    if method in {"kmeans", "logistic_regression"}:
                        fitter = create_fitter(method, random_state=config.seed)
                    elif method == "random":
                        fitter = create_fitter(method, layer=layer, seed=42)
                    else:
                        fitter = create_fitter(method)
                    result = fitter.fit(train_activations, labels_train)
                metadata = {
                    **result.diagnostics,
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "toy_train_examples": len(toy_train),
                    "model_revision": adapter.provenance()["resolved_model_revision"],
                    "tokenizer_revision": adapter.provenance()["resolved_tokenizer_revision"],
                    "prepend_bos": adapter.prepend_bos,
                }
                if result.direction.ndim == 1:
                    metadata["projection_threshold"] = projection_threshold(
                        train_activations, labels_train, result.direction
                    )
                artifact = DirectionArtifact(
                    method=method,
                    model_name=config.model.hub_name,
                    layer=layer,
                    vector=result.direction,
                    metadata=metadata,
                )
                artifact.save(path)
            if artifact.vector.ndim == 1:
                layer_directions[method] = artifact.vector
                threshold = float(
                    artifact.metadata.get(
                        "projection_threshold",
                        projection_threshold(train_activations, labels_train, artifact.vector),
                    )
                )
                toy_projection = projection_accuracy(
                    test_activations, labels_test, artifact.vector, threshold
                )
                sst_projection = projection_accuracy(
                    sst_activations, labels_sst, artifact.vector, threshold
                )
            else:
                threshold = float("nan")
                toy_projection = float("nan")
                sst_projection = float("nan")

            history = artifact.metadata.get("loss_history", [])
            for loss_row in history:
                das_loss_rows.append(
                    {
                        "model": config.model.name,
                        "method": method,
                        "layer": layer,
                        "subspace_dimension": artifact.metadata.get("subspace_dimension", 1),
                        "seed": config.seed,
                        "epochs": artifact.metadata.get("epochs"),
                        "learning_rate": config.das.learning_rate,
                        "batch_size": config.das.batch_size,
                        **loss_row,
                    }
                )
            if das_loss_rows:
                _write_rows(das_loss_rows, run_dir / "das_losses.csv")
            direction_metadata_rows.append(
                {
                    "model": config.model.name,
                    "method": method,
                    "layer": layer,
                    "subspace_dimension": artifact.metadata.get("subspace_dimension", 1),
                    "orientation_convention": artifact.metadata.get("orientation_convention"),
                    "orientation_reference": artifact.metadata.get("orientation_reference"),
                    "raw_orientation_dot": artifact.metadata.get("raw_orientation_dot"),
                    "orientation_sign_flipped": artifact.metadata.get("orientation_sign_flipped"),
                    "artifact_path": str(path),
                }
            )
            _write_rows(direction_metadata_rows, run_dir / "direction_metadata.csv")
            toy_patch = evaluate_directional_patching(
                adapter,
                test_pairs,
                artifact.vector,
                layer=layer,
                answers=toy.answers,
                position="all",
                batch_size=config.model.batch_size,
            )
            sst_patch = evaluate_directional_patching(
                adapter,
                sst_pairs,
                artifact.vector,
                layer=layer,
                answers=SST_CONTINUATION_ANSWERS,
                position="all",
                batch_size=config.model.batch_size,
            )
            for dataset_name, result in (("toy_test", toy_patch), ("sst", sst_patch)):
                patching_rows.extend(
                    {
                        "model": config.model.name,
                        "method": method,
                        "layer": layer,
                        "dataset": dataset_name,
                        **record,
                    }
                    for record in result.records
                )
            _write_rows(patching_rows, run_dir / "patching_records.csv")
            row = {
                "model": config.model.name,
                "method": method,
                "layer": layer,
                "subspace_dimension": artifact.metadata.get("subspace_dimension", 1),
                "toy_projection_accuracy": toy_projection,
                "toy_patch_recovery": toy_patch.recovery,
                "toy_flip_rate": toy_patch.flip_rate,
                "toy_sign_flip_rate": toy_patch.sign_flip_rate,
                "toy_logit_diff_percent": toy_patch.recovery_percent,
                "toy_logit_flip_percent": toy_patch.flip_percent,
                "toy_sign_flip_percent": toy_patch.sign_flip_percent,
                "toy_corrupted_accuracy": toy_patch.corrupted_accuracy,
                "toy_clean_accuracy": toy_patch.clean_accuracy,
                "toy_patched_accuracy": toy_patch.patched_accuracy,
                "sst_projection_accuracy": sst_projection,
                "sst_patch_recovery": sst_patch.recovery,
                "sst_flip_rate": sst_patch.flip_rate,
                "sst_sign_flip_rate": sst_patch.sign_flip_rate,
                "sst_logit_diff_percent": sst_patch.recovery_percent,
                "sst_logit_flip_percent": sst_patch.flip_percent,
                "sst_sign_flip_percent": sst_patch.sign_flip_percent,
                "sst_corrupted_accuracy": sst_patch.corrupted_accuracy,
                "sst_clean_accuracy": sst_patch.clean_accuracy,
                "sst_patched_accuracy": sst_patch.patched_accuracy,
                "n_toy_test": len(toy_test),
                "n_sst": len(sst_examples_for_projection),
                "n_sst_pairs": len(sst_pairs),
            }
            if owt_texts is not None:
                owt = evaluate_openwebtext_ablation(
                    adapter,
                    owt_texts,
                    artifact.vector,
                    layer=layer,
                    sequence_length=config.data.openwebtext_sequence_length,
                    batch_size=config.model.batch_size,
                    seed=config.seed,
                )
                row.update(
                    owt_baseline_loss=owt.baseline_loss,
                    owt_ablated_loss=owt.ablated_loss,
                    owt_loss_delta=owt.loss_delta,
                )
            metrics_rows.append(row)
            _write_rows(metrics_rows, run_dir / "metrics.csv")
            clear_device_cache(device_spec.device)

        similarities = cosine_similarity_table(layer_directions)
        for left in similarities.index:
            for right in similarities.columns:
                similarity_rows.append(
                    {
                        "model": config.model.name,
                        "layer": layer,
                        "method_a": left,
                        "method_b": right,
                        "absolute_cosine": similarities.loc[left, right],
                    }
                )
        _write_rows(similarity_rows, run_dir / "direction_similarities.csv")
        if owt_texts is not None:
            for random_seed in config.data.openwebtext_random_seeds:
                generator = np.random.default_rng(random_seed)
                random_direction = generator.normal(size=adapter.hidden_size)
                control = evaluate_openwebtext_ablation(
                    adapter,
                    owt_texts,
                    random_direction,
                    layer=layer,
                    sequence_length=config.data.openwebtext_sequence_length,
                    batch_size=config.model.batch_size,
                    seed=random_seed,
                )
                owt_control_rows.append(
                    {
                        "model": config.model.name,
                        "layer": layer,
                        "control": "random_direction",
                        "seed": random_seed,
                        "baseline_loss": control.baseline_loss,
                        "ablated_loss": control.ablated_loss,
                        "loss_delta": control.loss_delta,
                    }
                )
            _write_rows(owt_control_rows, run_dir / "openwebtext_controls.csv")

    metrics = pd.DataFrame(metrics_rows)
    selection = config.experiment.selection_metric
    if selection not in metrics.columns:
        raise ValueError(f"Selection metric {selection!r} not in {list(metrics.columns)}")
    valid = metrics.dropna(subset=[selection])
    best = valid.loc[valid.groupby("method")[selection].idxmax()].sort_values("method")
    best.to_csv(run_dir / "best_layers.csv", index=False)
    return run_dir
