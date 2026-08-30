from pathlib import Path

import numpy as np

from sentiment_manifold.artifacts import DirectionArtifact
from sentiment_manifold.config import ExperimentConfig, ReproductionConfig
from sentiment_manifold.data.toy_movie_review import load_toy_movie_review


PROJECT_ROOT = Path(__file__).parents[1]


def test_toy_movie_review_preserves_paper_split_counts():
    dataset = load_toy_movie_review(PROJECT_ROOT / "data/toy_movie_review.yaml")
    assert len(dataset.train) == 55
    assert len(dataset.test) == 30
    assert len(dataset.paired("train")) == 48
    assert len(dataset.paired("test")) == 28
    assert all(example.text.endswith("Conclusion: This movie is") for example in dataset.train)


def test_toy_prompt_and_answers_match_tigges_source_bytes():
    dataset = load_toy_movie_review(PROJECT_ROOT / "data/toy_movie_review.yaml")
    assert dataset.train[0].text == (
        "I thought this movie was adequate, I enjoyed it. \nConclusion: This movie is"
    )
    assert dataset.answers[1] == (" great", " amazing", " awesome", " good", " perfect")
    assert dataset.answers[0] == (
        " terrible",
        " awful",
        " bad",
        " horrible",
        " disgusting",
    )
    assert dataset.verbs[1] == ("enjoyed", "loved", "liked", "appreciated", "admired")
    assert dataset.verbs[0] == ("hated", "disliked", "despised")
    assert sum(len(words) for words in dataset.verbs.values()) == 8
    assert "extraordinary" in dataset.adjectives["train"][1]


def test_toy_corruption_uses_upstream_cyclic_shift():
    dataset = load_toy_movie_review(PROJECT_ROOT / "data/toy_movie_review.yaml")
    pairs = dataset.paired("test")
    assert pairs[0].clean.label == 1 and pairs[0].corrupted.label == 0
    assert pairs[1].clean.label == 0 and pairs[1].corrupted.label == 1
    assert pairs[1].corrupted.example_id.endswith("pos-001")
    assert pairs[-1].corrupted.example_id == pairs[0].clean.example_id


def test_direction_artifact_round_trip(tmp_path):
    artifact = DirectionArtifact("pca", "gpt2", 3, np.array([3.0, 4.0]), {"score": 1.0})
    path = artifact.save(tmp_path / "direction.npz")
    restored = DirectionArtifact.load(path)
    np.testing.assert_allclose(restored.vector, [0.6, 0.8])
    assert restored.metadata == {"score": 1.0}


def test_subspace_direction_artifact_round_trip(tmp_path):
    matrix = np.array([[2.0, 0.0], [0.0, -3.0], [0.0, 0.0]])
    artifact = DirectionArtifact("das2d", "gpt2", 3, matrix, {"subspace_dimension": 2})
    restored = DirectionArtifact.load(artifact.save(tmp_path / "subspace.npz"))
    assert restored.vector.shape == (3, 2)
    np.testing.assert_allclose(restored.vector.T @ restored.vector, np.eye(2), atol=1e-7)
    assert restored.vector[0, 0] > 0
    assert restored.vector[1, 1] < 0


def test_resample_ablation_is_explicitly_opt_in():
    config = ExperimentConfig()
    assert not config.resample_ablation_enabled
    config.openwebtext_resample_ablation = True
    assert config.resample_ablation_enabled


def test_legacy_openwebtext_flag_remains_compatible():
    config = ExperimentConfig(evaluate_openwebtext=True)
    assert config.resample_ablation_enabled


def test_reproduction_config_pins_gpt2_and_tigges_settings():
    config = ReproductionConfig.load(PROJECT_ROOT / "configs/reproduction.yaml")
    assert config.model.revision == "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    assert config.model.prepend_bos is True
    assert config.das.epochs == 64
    assert config.das.batch_size == 128
    assert config.das.implementation == "tigges_rotation"
    assert config.fitting.kmeans_n_init == 10
    assert config.fitting.logistic_c == 1.0
    assert config.fitting.logistic_solver == "liblinear"
    assert config.experiment.output_dir == str((PROJECT_ROOT / "outputs/results").resolve())
    assert config.experiment.checkpoint_dir == str((PROJECT_ROOT / "checkpoints").resolve())
    assert config.experiment.methods == [
        "mean_diff",
        "kmeans",
        "logistic_regression",
        "pca",
        "das",
        "das2d",
        "das3d",
        "random",
    ]
