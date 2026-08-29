import pytest
import torch

from sentiment_manifold.directions.das import directional_replace
from sentiment_manifold.evaluation.patching import (
    PatchingResult,
    _centered_target_signed_margins,
    _logit_differences,
    _target_directed_logit_flips,
    _target_signed_margins,
)


def test_directional_replace_preserves_orthogonal_component():
    corrupted = torch.tensor([[1.0, 5.0]])
    clean = torch.tensor([[3.0, -8.0]])
    direction = torch.tensor([1.0, 0.0])
    patched = directional_replace(corrupted, clean, direction)
    torch.testing.assert_close(patched, torch.tensor([[3.0, 5.0]]))


def test_directional_replace_supports_das_subspaces():
    corrupted = torch.tensor([[1.0, 2.0, 5.0]])
    clean = torch.tensor([[3.0, 4.0, -8.0]])
    basis = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    patched = directional_replace(corrupted, clean, basis)
    torch.testing.assert_close(patched, torch.tensor([[3.0, 4.0, 5.0]]))


def test_target_signed_margins_match_correct_minus_incorrect_answer_order():
    # Vocabulary positions 0 and 1 are the negative and positive answers.
    logits = torch.tensor([[4.0, 1.0], [2.0, 7.0]])
    differences = _logit_differences(logits, {0: 0, 1: 1})
    margins = _target_signed_margins(differences, torch.tensor([0, 1]))

    torch.testing.assert_close(differences, torch.tensor([-3.0, 5.0]))
    torch.testing.assert_close(margins, torch.tensor([3.0, 5.0]))


def test_logit_difference_averages_tigges_answer_pairs():
    logits = torch.tensor([[1.0, 2.0, 6.0, 10.0]])
    differences = _logit_differences(
        logits,
        {0: torch.tensor([0, 1]), 1: torch.tensor([2, 3])},
    )
    torch.testing.assert_close(differences, torch.tensor([6.5]))


def test_logit_flip_requires_pre_post_sign_change_toward_target():
    corrupted = torch.tensor([2.0, -3.0, -2.0, 1.0])
    patched = torch.tensor([-1.0, 4.0, -1.0, -2.0])
    targets = torch.tensor([0, 1, 1, 1])

    flips = _target_directed_logit_flips(corrupted, patched, targets)

    # First two cross zero toward the target. The third never crosses zero;
    # the fourth crosses away from a target the corrupted run already predicted.
    torch.testing.assert_close(flips, torch.tensor([True, True, False, False]))


def test_centered_target_margins_match_tigges_classifier_debiasing():
    differences = torch.tensor([5.0, -1.0, 3.0, -3.0])
    targets = torch.tensor([1, 0, 1, 0])

    centered = _centered_target_signed_margins(differences, targets)

    # Removing the +1 positive-minus-negative bias, followed by target
    # orientation, matches Tigges's center_logit_diffs helper.
    torch.testing.assert_close(centered, torch.tensor([4.0, 2.0, 2.0, 4.0]))


def test_patching_result_exposes_paper_percentage_scale():
    result = PatchingResult(
        recovery=1.098,
        flip_rate=0.535,
        sign_flip_rate=0.535,
        corrupted_margin=-1.0,
        clean_margin=1.0,
        patched_margin=1.196,
        corrupted_accuracy=0.0,
        clean_accuracy=1.0,
        patched_accuracy=0.535,
        n_pairs=1,
        records=(),
    )

    assert result.recovery_percent == pytest.approx(109.8)
    assert result.flip_percent == pytest.approx(53.5)
    assert result.sign_flip_percent == pytest.approx(53.5)
