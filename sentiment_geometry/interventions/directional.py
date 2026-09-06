"""Directional and subspace activation replacement operators."""

from __future__ import annotations

from torch import Tensor


def _orthonormal_basis(direction: Tensor) -> Tensor:
    basis = direction.float()
    if basis.ndim == 1:
        basis = basis.unsqueeze(1)
    if basis.ndim != 2:
        raise ValueError("Direction must have shape [d_model] or [d_model, d_subspace]")
    norms = basis.norm(dim=0, keepdim=True)
    if (norms <= 1e-12).any():
        raise ValueError("Direction/subspace contains a zero basis vector")
    return basis / norms


def directional_replace(corrupted: Tensor, clean: Tensor, direction: Tensor) -> Tensor:
    """Replace the corrupted projection onto a direction or subspace."""

    basis = _orthonormal_basis(direction).to(corrupted.dtype)
    coefficients = (clean - corrupted) @ basis
    return corrupted + coefficients @ basis.transpose(-2, -1)
