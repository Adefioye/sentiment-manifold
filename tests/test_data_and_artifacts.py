from pathlib import Path

import numpy as np

from sentiment_manifold.artifacts import DirectionArtifact
from sentiment_manifold.config import ExperimentConfig
from sentiment_manifold.data.toy_movie_review import load_toy_movie_review


PROJECT_ROOT = Path(__file__).parents[1]


def test_toy_movie_review_preserves_paper_split_counts():
    dataset = load_toy_movie_review(PROJECT_ROOT / "data/toy_movie_review.yaml")
    # The checked-in upstream prompts.yaml contains 54 train adjectives even
    # though the paper describes a 55/30 split. Preserve the source artifact.
    assert len(dataset.train) == 54
    assert len(dataset.test) == 30
    assert len(dataset.paired("train")) == 48
    assert len(dataset.paired("test")) == 28
    assert all(example.text.endswith("Conclusion: This movie is") for example in dataset.train)


def test_direction_artifact_round_trip(tmp_path):
    artifact = DirectionArtifact("pca", "gpt2", 3, np.array([3.0, 4.0]), {"score": 1.0})
    path = artifact.save(tmp_path / "direction.npz")
    restored = DirectionArtifact.load(path)
    np.testing.assert_allclose(restored.vector, [0.6, 0.8])
    assert restored.metadata == {"score": 1.0}


def test_resample_ablation_is_explicitly_opt_in():
    config = ExperimentConfig()
    assert not config.resample_ablation_enabled
    config.openwebtext_resample_ablation = True
    assert config.resample_ablation_enabled


def test_legacy_openwebtext_flag_remains_compatible():
    config = ExperimentConfig(evaluate_openwebtext=True)
    assert config.resample_ablation_enabled
