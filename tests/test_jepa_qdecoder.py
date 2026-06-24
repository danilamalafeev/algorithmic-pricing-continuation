from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.dqn_oracle_vs_qvictim import (
    clone_params,
    dqn_shared_jepa_qdecoder_train_step,
    dqn_shared_jepa_train_step,
    init_shared_jepa_params,
    init_shared_jepa_qdecoder_params,
    latent_variance_regularizer,
    normalize_victim_q_rows,
    oracle_dqn_forward_for_kind,
    q_decoder_metrics,
    q_decoder_loss,
    init_replay_buffer,
    replay_add,
    replay_sample,
    shared_jepa_qdecoder_forward,
    victim_decoder_target,
)


def _batch(B: int, obs_dim: int, K: int) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randn(B, obs_dim),
        "action": torch.randint(0, K, (B,)),
        "reward": torch.randn(B),
        "next_obs": torch.randn(B, obs_dim),
        "done": torch.zeros(B),
        "victim_q_target": normalize_victim_q_rows(torch.randn(B, K)),
    }


def test_qdecoder_shapes_and_scale_invariant_targets():
    B, obs_dim, K = 6, 8, 5
    params = init_shared_jepa_qdecoder_params(
        torch.Generator().manual_seed(1),
        obs_dim,
        hidden_dim=7,
        latent_dim=4,
        K=K,
    )
    q, decoded_q, latent = shared_jepa_qdecoder_forward(params, torch.randn(B, obs_dim))
    assert q.shape == (B, K)
    assert decoded_q.shape == (B, K)
    assert latent.shape == (B, 4)

    raw = torch.tensor([[1.0, 2.0, 4.0], [7.0, 7.0, 7.0]])
    normalized = normalize_victim_q_rows(raw)
    torch.testing.assert_close(normalized[0], normalize_victim_q_rows(raw * 13.0 + 9.0)[0])
    torch.testing.assert_close(normalized[1], torch.zeros(3))


def test_configurable_decoder_targets_are_correct():
    q = torch.tensor([[1.0, 3.0, 2.0], [4.0, 4.0, 1.0]])

    normalized = victim_decoder_target(q, "normalized_q_values")
    torch.testing.assert_close(normalized, normalize_victim_q_rows(q))
    advantages = victim_decoder_target(q, "centered_advantages")
    torch.testing.assert_close(advantages, torch.tensor([[-2.0, 0.0, -1.0], [0.0, 0.0, -3.0]]))
    classification = victim_decoder_target(q, "greedy_action_classification")
    torch.testing.assert_close(classification, torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))
    delta = torch.tensor([[0.0, 0.5, 0.0], [-0.25, 0.0, 0.0]])
    delta_target = victim_decoder_target(q, "normalized_q_delta", q_delta_rows=delta)
    torch.testing.assert_close(delta_target, normalize_victim_q_rows(delta))


def test_normalized_q_delta_requires_matching_delta_rows():
    q = torch.ones(2, 3)
    with pytest.raises(ValueError, match="requires q_delta_rows"):
        victim_decoder_target(q, "normalized_q_delta")
    with pytest.raises(ValueError, match="must have shape"):
        victim_decoder_target(q, "normalized_q_delta", q_delta_rows=torch.ones(2, 2))


def test_classification_target_uses_cross_entropy():
    logits = torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
    target = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 0]))
    torch.testing.assert_close(q_decoder_loss(logits, target, "greedy_action_classification"), expected)


def test_classification_loss_supports_inverse_frequency_balancing():
    logits = torch.tensor(
        [[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
        requires_grad=True,
    )
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 0, 1])
    expected = torch.nn.functional.cross_entropy(logits, labels, weight=torch.tensor([0.5, 1.5]))
    actual = q_decoder_loss(
        logits,
        target,
        "greedy_action_classification",
        class_balance=True,
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.count_nonzero(logits.grad).item() > 0


def test_replay_sample_can_limit_q_targets_by_age():
    B, obs_dim, K = 6, 4, 3
    buffer = init_replay_buffer(capacity=10, obs_dim=obs_dim, K=K)
    replay_add(
        buffer,
        torch.randn(B, obs_dim),
        torch.zeros(B, dtype=torch.long),
        torch.zeros(B),
        torch.randn(B, obs_dim),
        torch.zeros(B, dtype=torch.bool),
        victim_action=torch.zeros(B, dtype=torch.long),
        cf_profit=torch.zeros(B, K),
        target_step=torch.arange(B),
    )
    batch = replay_sample(
        buffer,
        batch_size=20,
        generator=torch.Generator().manual_seed(9),
        current_step=5,
        max_target_age=2,
    )
    assert torch.all(batch["target_step"] >= 3)


def test_qdecoder_gradients_reach_shared_encoder():
    B, obs_dim, K = 8, 6, 5
    params = init_shared_jepa_qdecoder_params(
        torch.Generator().manual_seed(2),
        obs_dim,
        hidden_dim=7,
        latent_dim=4,
        K=K,
    )
    _, decoded_q, _ = shared_jepa_qdecoder_forward(params, torch.randn(B, obs_dim))
    loss = torch.mean((decoded_q - torch.randn(B, K)) ** 2)
    loss.backward()
    assert params["enc_W1"].grad is not None
    assert torch.count_nonzero(params["enc_W1"].grad).item() > 0


def test_decoder_target_is_not_part_of_action_selection_input():
    B, obs_dim, K = 4, 6, 5
    params = init_shared_jepa_qdecoder_params(
        torch.Generator().manual_seed(3),
        obs_dim,
        hidden_dim=7,
        latent_dim=4,
        K=K,
    )
    obs = torch.randn(B, obs_dim)
    q_before = oracle_dqn_forward_for_kind("dqn_shared_jepa_qdecoder", params, obs)
    unrelated_target = normalize_victim_q_rows(torch.randn(B, K) * 1000.0)
    q_after = oracle_dqn_forward_for_kind("dqn_shared_jepa_qdecoder", params, obs)
    assert unrelated_target.shape == (B, K)
    torch.testing.assert_close(q_before, q_after)


@pytest.mark.parametrize(
    "target_mode",
    [
        "normalized_q_values",
        "centered_advantages",
        "greedy_action_classification",
        "normalized_q_delta",
    ],
)
def test_zero_decoder_coefficient_matches_shared_jepa_training_path(target_mode):
    B, obs_dim, K = 8, 6, 5
    base = init_shared_jepa_params(
        torch.Generator().manual_seed(4),
        obs_dim,
        hidden_dim=7,
        latent_dim=4,
        K=K,
    )
    decoded = init_shared_jepa_qdecoder_params(
        torch.Generator().manual_seed(4),
        obs_dim,
        hidden_dim=7,
        latent_dim=4,
        K=K,
    )
    batch = _batch(B, obs_dim, K)
    base_target = clone_params(base)
    decoded_target = clone_params(decoded)
    base_optimizer = torch.optim.SGD(list(base.values()), lr=1.0e-2)
    decoded_optimizer = torch.optim.SGD(list(decoded.values()), lr=1.0e-2)
    base_metrics = dqn_shared_jepa_train_step(
        base,
        base_target,
        batch,
        base_optimizer,
        gamma=0.95,
        jepa_coef=0.1,
        K=K,
    )
    decoded_metrics = dqn_shared_jepa_qdecoder_train_step(
        decoded,
        decoded_target,
        batch,
        decoded_optimizer,
        gamma=0.95,
        jepa_coef=0.1,
        q_decoder_coef=0.0,
        collapse_coef=100.0,
        collapse_target_std=1.0,
        K=K,
        q_decoder_target=target_mode,
    )
    assert decoded_metrics["total_loss"] == pytest.approx(base_metrics["total_loss"])
    for key in base:
        torch.testing.assert_close(decoded[key], base[key])


def test_qdecoder_metrics_are_correct():
    target = torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
    predicted = torch.tensor([[0.0, 1.5, 1.0], [1.0, 2.0, 0.0]])
    metrics = q_decoder_metrics(predicted, target)
    assert metrics["normalized_q_mse"].item() == pytest.approx(9.25 / 6.0)
    assert metrics["greedy_action_accuracy"].item() == pytest.approx(0.5)
    assert metrics["q_gap_prediction_error"].item() == pytest.approx(0.25)


def test_collapse_regularizer_penalizes_low_variance_latents():
    collapsed = torch.zeros(16, 4)
    spread = torch.tensor(
        [[-1.0, 1.0, -1.0, 1.0], [1.0, -1.0, 1.0, -1.0]],
        dtype=torch.float32,
    ).repeat(8, 1)
    collapsed_loss, collapsed_metrics = latent_variance_regularizer(collapsed, target_std=0.5)
    spread_loss, spread_metrics = latent_variance_regularizer(spread, target_std=0.5)
    assert collapsed_loss.item() > 0.0
    assert spread_loss.item() == pytest.approx(0.0)
    assert collapsed_metrics["latent_feature_variance"].item() == pytest.approx(0.0)
    assert spread_metrics["latent_feature_variance"].item() > 0.0
    assert collapsed_metrics["latent_collapsed_fraction"].item() == pytest.approx(1.0)
    assert spread_metrics["latent_collapsed_fraction"].item() == pytest.approx(0.0)


def test_decoder_metrics_include_baselines_replay_age_and_scale_invariant_diagnostics():
    B, obs_dim, K = 8, 6, 5
    params = init_shared_jepa_qdecoder_params(
        torch.Generator().manual_seed(8),
        obs_dim,
        hidden_dim=7,
        latent_dim=4,
        K=K,
    )
    target_params = clone_params(params)
    batch = _batch(B, obs_dim, K)
    batch["target_step"] = torch.arange(B)
    optimizer = torch.optim.SGD(list(params.values()), lr=1.0e-2)
    metrics = dqn_shared_jepa_qdecoder_train_step(
        params,
        target_params,
        batch,
        optimizer,
        gamma=0.95,
        jepa_coef=0.1,
        q_decoder_coef=0.1,
        collapse_coef=0.0,
        collapse_target_std=0.1,
        K=K,
        diagnostics_state={},
        current_step=10,
    )
    expected = {
        "zero_q_baseline_mse",
        "running_mean_q_baseline_mse",
        "majority_greedy_accuracy",
        "decoder_mse_improvement_over_zero",
        "decoder_mse_improvement_over_running_mean",
        "decoder_accuracy_improvement_over_majority",
        "greedy_target_entropy",
        "q_target_gap_mean",
        "replay_target_age_mean",
        "replay_target_age_max",
        "latent_effective_covariance_rank",
        "latent_participation_ratio",
        "latent_mean_abs_offdiag_correlation",
        "latent_normalized_variance_min",
        "latent_normalized_variance_max",
        "latent_normalized_variance_std",
    }
    assert expected.issubset(metrics)
    assert metrics["replay_target_age_mean"] == pytest.approx(6.5)
    assert metrics["replay_target_age_max"] == pytest.approx(10.0)
    assert all(np.isfinite(metrics[key]) for key in expected)
