import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sentiment_geometry.analysis import (
    DirectionAlignmentConfig,
    DirectionAlignmentSource,
    PairwiseDirectionAlignmentConfig,
    run_direction_alignment_analysis,
    run_pairwise_direction_alignment_analysis,
)
from sentiment_geometry.fitting_methods import DirectionArtifact
from sentiment_geometry.models import ModelConfig
from sentiment_geometry.reporting import (
    plot_direction_alignment,
    plot_pairwise_absolute_direction_alignment,
)


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/sentiment_valence_direction_alignment.yaml"
PAIRWISE_CONFIG_PATH = (
    PROJECT_ROOT / "configs/ait_training_scope_direction_alignment.yaml"
)


def _write_source(
    storage_root: Path,
    source: DirectionAlignmentSource,
    vectors: dict[str, np.ndarray],
    layers: dict[str, int],
    *,
    domain: str | None = None,
    activation_representation: str | None = None,
    manifest_status: str = "completed",
) -> None:
    root = source.run_root(storage_root)
    model_root = root / "results" / "gpt2-small"
    directions_root = root / "directions"
    model_root.mkdir(parents=True)
    directions_root.mkdir(parents=True)
    (root / "run_manifest.json").write_text(
        json.dumps({"run_id": source.run_id, "status": manifest_status}),
        encoding="utf-8",
    )
    selection_rows = []
    metadata_rows = []
    selected_metric_rows = []
    for method, vector in vectors.items():
        layer = layers[method]
        relative = Path("gpt2-small") / method / f"layer{layer:02d}.npz"
        checkpoint = directions_root / relative
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "artifact_schema_version": 2,
            "fit_position": source.fit_position,
            "model_revision": "model-commit",
            "tokenizer_revision": "model-commit",
            "orientation_convention": "negative_to_positive",
        }
        if domain is not None:
            metadata["domain"] = domain
        if activation_representation is not None:
            metadata["activation_representation"] = activation_representation
        DirectionArtifact(
            method=method,
            model_name="gpt2",
            layer=layer,
            vector=vector,
            metadata=metadata,
        ).save(checkpoint)
        selection_rows.append(
            {
                "model": "gpt2-small",
                "method": method,
                "fit_position": source.fit_position,
                "selected_layer": layer,
                "selection_dataset": source.selection_dataset,
                "selection_metric": source.selection_metric,
                "selection_value_percent": 75.0,
                "direction_checkpoint": "/stale/path/checkpoint.npz",
            }
        )
        metadata_rows.append(
            {
                "model": "gpt2-small",
                "method": method,
                "fit_position": source.fit_position,
                "layer": layer,
                "artifact_path": "/stale/path/checkpoint.npz",
                "artifact_relative_path": str(relative),
            }
        )
        selected_metric_rows.append(
            {
                "model": "gpt2-small",
                "method": method,
                "fit_position": source.fit_position,
                "dataset": source.evaluation_dataset,
                "layer": layer,
            }
        )
    pd.DataFrame(selection_rows).to_csv(model_root / "layer_selection.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(model_root / "direction_metadata.csv", index=False)
    pd.DataFrame(selected_metric_rows).to_csv(
        model_root / "selected_metrics.csv", index=False
    )


def test_alignment_config_pins_both_completed_runs_and_methods():
    config = DirectionAlignmentConfig.load(CONFIG_PATH)

    assert config.output_experiment_name == "sentiment-valence-direction-alignment"
    assert config.methods == ["mean_diff", "das"]
    sources = {source.name: source for source in config.sources}
    assert sources["sentiment"].run_id == "2026-09-18_20-22_CDT"
    assert sources["sentiment"].selection_dataset == "toy_adverbs"
    assert sources["valence"].run_id == "2026-09-23_09-10_CDT"
    assert sources["valence"].selection_dataset == "ait_eval"
    assert sources["valence"].expected_activation_representation == "last_token"


def test_direction_alignment_loads_selected_artifacts_and_computes_three_views(tmp_path):
    sources = [
        DirectionAlignmentSource(
            name="sentiment",
            experiment_name="sentiment-position-comparison",
            run_id="sentiment-run",
            fit_position="final",
            selection_dataset="toy_adverbs",
            selection_metric="logit_flip_percent",
            evaluation_dataset="sst",
        ),
        DirectionAlignmentSource(
            name="valence",
            experiment_name="full-ait-last-token-directions",
            run_id="valence-run",
            fit_position="final",
            selection_dataset="ait_eval",
            selection_metric="logit_flip_percent",
            evaluation_dataset="ait_test",
            expected_domain="ait_valence",
            expected_activation_representation="last_token",
        ),
    ]
    _write_source(
        tmp_path,
        sources[0],
        {"mean_diff": np.array([1.0, 0.0]), "das": np.array([0.0, 1.0])},
        {"mean_diff": 3, "das": 4},
    )
    root_two = float(np.sqrt(2.0))
    _write_source(
        tmp_path,
        sources[1],
        {
            "mean_diff": np.array([1.0, 0.0]),
            "das": np.array([1.0 / root_two, 1.0 / root_two]),
        },
        {"mean_diff": 5, "das": 6},
        domain="ait_valence",
        activation_representation="last_token",
        manifest_status="resumed",
    )
    config = DirectionAlignmentConfig(
        output_experiment_name="alignment",
        models=[ModelConfig(name="gpt2-small", revision="model-commit")],
        sources=sources,
        methods=["mean_diff", "das"],
    )

    result = run_direction_alignment_analysis(
        config,
        storage_root=tmp_path,
        output_dir=tmp_path / "output",
    )

    assert len(result.selected_directions) == 4
    assert len(result.similarities) == 12
    assert set(result.similarities["comparison"]) == {
        "within_sentiment",
        "within_valence",
        "valence_vs_sentiment",
    }
    mean_alignment = result.same_method_alignment[
        result.same_method_alignment["method"] == "mean_diff"
    ].iloc[0]
    das_alignment = result.same_method_alignment[
        result.same_method_alignment["method"] == "das"
    ].iloc[0]
    assert np.isclose(mean_alignment["signed_cosine"], 1.0)
    assert np.isclose(das_alignment["signed_cosine"], 1.0 / root_two)
    for filename in (
        "selected_directions.csv",
        "direction_cosines.csv",
        "same_method_cross_alignment.csv",
        "resolved_config.json",
    ):
        assert (result.output_dir / filename).is_file()

    paths = plot_direction_alignment(
        result.similarities,
        result.selected_directions,
        figure_dir=tmp_path / "figures",
        model_names=["gpt2-small"],
        methods=config.methods,
    )
    assert len(paths) == 6
    assert all(path.is_file() for path in paths)
    plt.close("all")


def test_pairwise_alignment_config_pins_partial_and_full_ait_runs():
    config = PairwiseDirectionAlignmentConfig.load(PAIRWISE_CONFIG_PATH)

    assert config.output_experiment_name == "ait-training-scope-direction-alignment"
    assert config.row_source == "full_ait"
    assert config.column_source == "partial_ait"
    assert config.methods == ["mean_diff", "das"]
    sources = {source.name: source for source in config.sources}
    assert sources["partial_ait"].experiment_name == "ait-last-token-directions"
    assert sources["partial_ait"].run_id == "2026-09-23_02-47_CDT"
    assert sources["partial_ait"].selection_dataset == "ait_test"
    assert sources["full_ait"].experiment_name == "full-ait-last-token-directions"
    assert sources["full_ait"].run_id == "2026-09-23_09-10_CDT"
    assert sources["full_ait"].selection_dataset == "ait_eval"


def test_pairwise_alignment_computes_only_cross_source_absolute_cosines(tmp_path):
    sources = [
        DirectionAlignmentSource(
            name="partial_ait",
            experiment_name="ait-last-token-directions",
            run_id="partial-run",
            fit_position="final",
            selection_dataset="ait_test",
            selection_metric="logit_flip_percent",
            evaluation_dataset="ait_test",
            expected_domain="ait_valence",
            expected_activation_representation="last_token",
        ),
        DirectionAlignmentSource(
            name="full_ait",
            experiment_name="full-ait-last-token-directions",
            run_id="full-run",
            fit_position="final",
            selection_dataset="ait_eval",
            selection_metric="logit_flip_percent",
            evaluation_dataset="ait_test",
            expected_domain="ait_valence",
            expected_activation_representation="last_token",
        ),
    ]
    _write_source(
        tmp_path,
        sources[0],
        {"mean_diff": np.array([1.0, 0.0]), "das": np.array([0.0, 1.0])},
        {"mean_diff": 3, "das": 4},
        domain="ait_valence",
        activation_representation="last_token",
    )
    root_two = float(np.sqrt(2.0))
    _write_source(
        tmp_path,
        sources[1],
        {
            "mean_diff": np.array([-1.0, 0.0]),
            "das": np.array([1.0 / root_two, 1.0 / root_two]),
        },
        {"mean_diff": 5, "das": 6},
        domain="ait_valence",
        activation_representation="last_token",
    )
    config = PairwiseDirectionAlignmentConfig(
        output_experiment_name="alignment",
        models=[ModelConfig(name="gpt2-small", revision="model-commit")],
        sources=sources,
        row_source="full_ait",
        column_source="partial_ait",
        methods=["mean_diff", "das"],
    )

    result = run_pairwise_direction_alignment_analysis(
        config,
        storage_root=tmp_path,
        output_dir=tmp_path / "output",
    )

    assert len(result.selected_directions) == 4
    assert "selection_value_percent" not in result.selected_directions
    assert list(result.absolute_cosines.columns) == [
        "model",
        "row_source",
        "row_method",
        "row_layer",
        "column_source",
        "column_method",
        "column_layer",
        "absolute_cosine",
    ]
    assert "signed_cosine" not in result.absolute_cosines
    matrix = result.absolute_cosines.pivot(
        index="row_method", columns="column_method", values="absolute_cosine"
    )
    assert np.isclose(matrix.loc["mean_diff", "mean_diff"], 1.0)
    assert np.isclose(matrix.loc["das", "mean_diff"], 1.0 / root_two)
    for filename in (
        "selected_direction_audit.csv",
        "absolute_cosines.csv",
        "resolved_config.json",
    ):
        assert (result.output_dir / filename).is_file()

    path = plot_pairwise_absolute_direction_alignment(
        result.absolute_cosines,
        figure_dir=tmp_path / "figures",
        model_names=["gpt2-small"],
        methods=config.methods,
        row_label="Full AIT",
        column_label="Partial AIT",
    )
    assert path.name == "absolute_cosine_full-ait_vs_partial-ait.png"
    assert path.is_file()
    plt.close("all")
