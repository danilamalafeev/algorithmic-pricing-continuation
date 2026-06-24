from __future__ import annotations

import pytest
import torch

from experiments.representation_diagnostics import (
    accuracy_improvement_over_baseline,
    effective_covariance_rank,
    init_running_mean_state,
    majority_greedy_accuracy,
    mean_absolute_off_diagonal_correlation,
    mse_improvement_over_baseline,
    participation_ratio,
    per_feature_normalized_variance,
    running_mean_mse,
    update_running_mean,
    zero_mse_baseline,
)


def test_zero_mse_baseline_matches_manual_value():
    target = torch.tensor([[1.0, -2.0], [3.0, 0.0]])
    assert zero_mse_baseline(target).item() == pytest.approx(3.5)


def test_running_mean_update_is_batch_partition_invariant():
    observations = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [8.0, 10.0]], dtype=torch.float64
    )
    one_batch = update_running_mean(
        init_running_mean_state(2, dtype=torch.float64), observations
    )
    split_batch = update_running_mean(
        update_running_mean(
            init_running_mean_state(2, dtype=torch.float64), observations[:1]
        ),
        observations[1:],
    )
    assert one_batch.count == 3
    torch.testing.assert_close(one_batch.mean, observations.mean(dim=0))
    torch.testing.assert_close(split_batch.mean, one_batch.mean)


def test_running_mean_mse_uses_historical_mean():
    state = update_running_mean(
        init_running_mean_state(2),
        torch.tensor([[1.0, 3.0], [3.0, 5.0]]),
    )
    target = torch.tensor([[2.0, 6.0], [4.0, 2.0]])
    assert running_mean_mse(state, target).item() == pytest.approx(3.0)


def test_majority_greedy_accuracy_counts_target_classes():
    target_q = torch.tensor(
        [[4.0, 1.0, 0.0], [2.0, 3.0, 1.0], [5.0, 0.0, 1.0], [7.0, 2.0, 1.0]]
    )
    assert majority_greedy_accuracy(target_q).item() == pytest.approx(0.75)


def test_relative_improvements_have_expected_sign_and_zero_behavior():
    assert mse_improvement_over_baseline(
        torch.tensor(0.25), torch.tensor(1.0)
    ).item() == pytest.approx(0.75)
    assert mse_improvement_over_baseline(
        torch.tensor(0.0), torch.tensor(0.0)
    ).item() == pytest.approx(0.0)
    assert accuracy_improvement_over_baseline(
        torch.tensor(0.8), torch.tensor(0.6)
    ).item() == pytest.approx(0.2)


def test_isotropic_latent_has_full_effective_rank_and_participation_ratio():
    latent = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=torch.float64,
    )
    assert effective_covariance_rank(latent).item() == pytest.approx(2.0)
    assert participation_ratio(latent).item() == pytest.approx(2.0)
    assert mean_absolute_off_diagonal_correlation(latent).item() == pytest.approx(0.0)
    torch.testing.assert_close(
        per_feature_normalized_variance(latent),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )


def test_rank_one_latent_metrics_detect_redundant_features():
    latent = torch.tensor(
        [[-2.0, -4.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 4.0]]
    )
    assert effective_covariance_rank(latent).item() == pytest.approx(1.0)
    assert participation_ratio(latent).item() == pytest.approx(1.0)
    assert mean_absolute_off_diagonal_correlation(latent).item() == pytest.approx(1.0)
    torch.testing.assert_close(
        per_feature_normalized_variance(latent), torch.tensor([0.2, 0.8])
    )


def test_latent_metrics_are_invariant_to_global_scale_and_offset():
    generator = torch.Generator().manual_seed(7)
    latent = torch.randn(32, 5, generator=generator, dtype=torch.float64)
    transformed = latent * 17.0 + 23.0
    torch.testing.assert_close(
        effective_covariance_rank(transformed), effective_covariance_rank(latent)
    )
    torch.testing.assert_close(
        participation_ratio(transformed), participation_ratio(latent)
    )
    torch.testing.assert_close(
        mean_absolute_off_diagonal_correlation(transformed),
        mean_absolute_off_diagonal_correlation(latent),
    )
    torch.testing.assert_close(
        per_feature_normalized_variance(transformed),
        per_feature_normalized_variance(latent),
    )


def test_degenerate_latent_metrics_are_finite_zeros():
    latent = torch.ones(8, 3)
    assert effective_covariance_rank(latent).item() == 0.0
    assert participation_ratio(latent).item() == 0.0
    assert mean_absolute_off_diagonal_correlation(latent).item() == 0.0
    torch.testing.assert_close(
        per_feature_normalized_variance(latent), torch.zeros(3)
    )


@pytest.mark.parametrize(
    ("function", "value"),
    [
        (zero_mse_baseline, torch.ones(4)),
        (majority_greedy_accuracy, torch.ones(2, 0)),
        (effective_covariance_rank, torch.ones(2, 3, 1)),
        (participation_ratio, torch.tensor([[1, 2], [3, 4]])),
    ],
)
def test_matrix_metrics_validate_shapes_and_dtype(function, value):
    with pytest.raises((TypeError, ValueError)):
        function(value)


def test_running_mean_validates_state_and_feature_shape():
    empty = init_running_mean_state(2)
    with pytest.raises(ValueError, match="at least one"):
        running_mean_mse(empty, torch.ones(2, 2))
    with pytest.raises(ValueError, match="feature dimension"):
        update_running_mean(empty, torch.ones(2, 3))
