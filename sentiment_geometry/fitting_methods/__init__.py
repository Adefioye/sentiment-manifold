from .artifacts import DirectionArtifact
from .base import DirectionFitter, FitResult
from .config import DASConfig, FittingConfig
from .linear import (
    KMeansFitter,
    LogisticRegressionFitter,
    MeanDifferenceFitter,
    PCAFitter,
    RandomDirectionFitter,
)

_FITTERS = {
    "mean_diff": MeanDifferenceFitter,
    "kmeans": KMeansFitter,
    "logistic_regression": LogisticRegressionFitter,
    "pca": PCAFitter,
    "random": RandomDirectionFitter,
}


def create_fitter(name: str, **kwargs) -> DirectionFitter:
    try:
        return _FITTERS[name](**kwargs)
    except KeyError as exc:
        if name in {"das", "das2d", "das3d"}:
            raise ValueError(
                "DAS is model-aware; construct DASFitter from fitting_methods.das"
            ) from exc
        raise ValueError(f"Unknown fitter {name!r}; available: {list_fitters()}") from exc


def list_fitters() -> list[str]:
    return [*_FITTERS, "das", "das2d", "das3d"]


__all__ = [
    "DASConfig",
    "DirectionArtifact",
    "DirectionFitter",
    "FitResult",
    "FittingConfig",
    "KMeansFitter",
    "LogisticRegressionFitter",
    "MeanDifferenceFitter",
    "PCAFitter",
    "RandomDirectionFitter",
    "create_fitter",
    "list_fitters",
]
