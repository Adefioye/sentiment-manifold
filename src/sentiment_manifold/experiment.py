"""End-to-end all-layer reproduction orchestration."""

from __future__ import annotations

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


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")


def _write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _filter_single_focus(adapter: CausalLMAdapter, examples):
    return [example for example in examples if adapter.focus_is_single_token(example)]


def run_reproduction(config: ReproductionConfig) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device_spec = resolve_device(config.model.device, config.model.dtype)
    adapter = CausalLMAdapter.from_pretrained(
        config.model.hub_name,
        device_spec,
        revision=config.model.revision,
    )
    run_dir = Path(config.experiment.output_dir) / _slug(config.model.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(config.to_dict(), indent=2))

    toy = load_toy_movie_review(config.data.toy_config)
    toy_train = _filter_single_focus(adapter, toy.train)
    toy_test = _filter_single_focus(adapter, toy.test)
    allowed_train = {example.example_id for example in toy_train}
    allowed_test = {example.example_id for example in toy_test}
    train_pairs = [
        pair
        for pair in toy.paired("train")
        if pair.base.example_id in allowed_train and pair.source.example_id in allowed_train
    ]
    test_pairs = [
        pair
        for pair in toy.paired("test")
        if pair.base.example_id in allowed_test and pair.source.example_id in allowed_test
    ]
    if not toy_train or not toy_test:
        raise RuntimeError("Tokenizer filtering removed every ToyMovieReview example")

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
            else:
                if method == "das":
                    fitter = DASFitter(
                        DASTrainingConfig(
                            epochs=config.das.epochs,
                            learning_rate=config.das.learning_rate,
                            weight_decay=config.das.weight_decay,
                            batch_size=config.das.batch_size,
                            max_grad_norm=config.das.max_grad_norm,
                            seed=config.seed,
                        )
                    )
                    result = fitter.fit(adapter, train_pairs, layer=layer, answers=toy.answers)
                else:
                    result = (
                        create_fitter(method, random_state=config.seed).fit(
                            train_activations, labels_train
                        )
                        if method in {"kmeans", "logistic_regression"}
                        else create_fitter(method).fit(train_activations, labels_train)
                    )
                threshold = projection_threshold(train_activations, labels_train, result.direction)
                artifact = DirectionArtifact(
                    method=method,
                    model_name=config.model.hub_name,
                    layer=layer,
                    vector=result.direction,
                    metadata={
                        **result.diagnostics,
                        "projection_threshold": threshold,
                        "toy_train_examples": len(toy_train),
                    },
                )
                artifact.save(path)
            layer_directions[method] = artifact.vector
            threshold = float(
                artifact.metadata.get(
                    "projection_threshold",
                    projection_threshold(train_activations, labels_train, artifact.vector),
                )
            )
            toy_patch = evaluate_directional_patching(
                adapter,
                test_pairs,
                artifact.vector,
                layer=layer,
                answers=toy.answers,
                position="focus",
                batch_size=config.model.batch_size,
            )
            sst_patch = evaluate_directional_patching(
                adapter,
                sst_pairs,
                artifact.vector,
                layer=layer,
                answers=toy.answers,
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
                "toy_projection_accuracy": projection_accuracy(
                    test_activations, labels_test, artifact.vector, threshold
                ),
                "toy_patch_recovery": toy_patch.recovery,
                "toy_flip_rate": toy_patch.flip_rate,
                "sst_projection_accuracy": projection_accuracy(
                    sst_activations, labels_sst, artifact.vector, threshold
                ),
                "sst_patch_recovery": sst_patch.recovery,
                "sst_flip_rate": sst_patch.flip_rate,
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
