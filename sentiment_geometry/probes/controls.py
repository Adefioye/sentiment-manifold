"""Example-level random-label controls with exact class-count preservation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def balanced_label_permutation(
    labels: NDArray[np.integer] | list[int], *, seed: int
) -> NDArray[np.int64]:
    """Permute binary labels without changing the split's class counts.

    This is deliberately an example-level random-label task, not Hewitt and
    Liang's type-level control task.  A separate permutation should be stored
    for every dataset split and experimental seed.
    """

    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(values) < 2 or set(np.unique(values)) != {0, 1}:
        raise ValueError("Random-label controls require both binary classes")
    return np.random.default_rng(seed).permutation(values).astype(np.int64, copy=False)


__all__ = ["balanced_label_permutation"]
