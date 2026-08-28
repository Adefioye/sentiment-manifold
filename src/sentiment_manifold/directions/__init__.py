from .base import DirectionFitter, FitResult
from .linear import KMeansFitter, LogisticRegressionFitter, MeanDifferenceFitter, PCAFitter


_FITTERS = {
    "mean_diff": MeanDifferenceFitter,
    "kmeans": KMeansFitter,
    "logistic_regression": LogisticRegressionFitter,
    "pca": PCAFitter,
}


def create_fitter(name: str, **kwargs) -> DirectionFitter:
    try:
        return _FITTERS[name](**kwargs)
    except KeyError as exc:
        if name == "das":
            raise ValueError("DAS is model-aware; construct DASFitter from directions.das") from exc
        raise ValueError(f"Unknown fitter {name!r}; available: {list_fitters()}") from exc


def list_fitters() -> list[str]:
    return [*_FITTERS, "das"]


__all__ = [
    "DirectionFitter",
    "FitResult",
    "KMeansFitter",
    "LogisticRegressionFitter",
    "MeanDifferenceFitter",
    "PCAFitter",
    "create_fitter",
    "list_fitters",
]
