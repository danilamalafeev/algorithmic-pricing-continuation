from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.oracle_rollout_lola import (
    tabular_rollout_lola_select_actions,
    tabular_rollout_lola_values,
    tabular_rollout_lola_values_torch,
)


K = 3
B = 2
PRICE_GRID = np.array([1.0, 2.0, 3.0], dtype=np.float64)


def _fixed_inputs() -> tuple[dict[str, np.ndarray], np.ndarray]:
    q = np.zeros((B, K * K, K), dtype=np.float64)
    for batch in range(B):
        for state in range(K * K):
            q[batch, state] = np.array([0.1, 0.5, 0.2]) + 0.01 * batch + 0.001 * state

    profit_matrix = np.zeros((K, K, 2), dtype=np.float64)
    for oracle_action in range(K):
        for victim_action in range(K):
            profit_matrix[oracle_action, victim_action, 0] = (
                2.0 * oracle_action - victim_action + 0.25 * oracle_action * victim_action
            )
            profit_matrix[oracle_action, victim_action, 1] = (
                1.5 * victim_action - 0.4 * oracle_action + 0.1 * oracle_action * victim_action
            )

    victim = {
        "Q": q,
        "state_id": np.array([0, 4], dtype=np.int64),
        "t": np.array([10, 11], dtype=np.int64),
    }
    return victim, profit_matrix


def _numpy_values(
    *,
    victim_policy_mode: str = "greedy",
    oracle_rollout_policy: str = "fixed_first_action",
    include_immediate: bool = True,
    seed: int = 7,
) -> tuple[np.ndarray, dict[str, float]]:
    victim, profit_matrix = _fixed_inputs()
    return tabular_rollout_lola_values(
        victim=victim,
        profit_matrix=profit_matrix,
        K=K,
        alpha=0.2,
        delta=0.9,
        beta=100.0,
        horizon=3,
        num_particles=2,
        victim_policy_mode=victim_policy_mode,
        oracle_rollout_policy=oracle_rollout_policy,
        discount=0.8,
        include_immediate=include_immediate,
        rng=np.random.default_rng(seed),
        price_grid=PRICE_GRID,
    )


def _torch_values(
    *,
    victim_policy_mode: str = "greedy",
    oracle_rollout_policy: str = "fixed_first_action",
    include_immediate: bool = True,
    seed: int = 7,
) -> tuple[np.ndarray, dict[str, float]]:
    victim, profit_matrix = _fixed_inputs()
    values, metrics = tabular_rollout_lola_values_torch(
        victim=victim,
        profit_matrix=profit_matrix,
        K=K,
        alpha=0.2,
        delta=0.9,
        beta=100.0,
        horizon=3,
        num_particles=2,
        victim_policy_mode=victim_policy_mode,
        oracle_rollout_policy=oracle_rollout_policy,
        discount=0.8,
        include_immediate=include_immediate,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        device=torch.device("cpu"),
        price_grid=PRICE_GRID,
        chunk_size=2,
    )
    return values.detach().cpu().numpy(), metrics


@pytest.mark.parametrize(
    ("oracle_rollout_policy", "expected"),
    [
        (
            "fixed_first_action",
            np.array([[-2.44, 3.05, 8.54], [-2.44, 3.05, 8.54]]),
        ),
        (
            "greedy_best_response",
            np.array([[4.04, 6.29, 8.54], [4.04, 6.29, 8.54]]),
        ),
    ],
)
def test_numpy_torch_candidate_values_match_fixed_regression(
    oracle_rollout_policy: str,
    expected: np.ndarray,
) -> None:
    numpy_values, numpy_metrics = _numpy_values(oracle_rollout_policy=oracle_rollout_policy)
    torch_values, torch_metrics = _torch_values(oracle_rollout_policy=oracle_rollout_policy)

    np.testing.assert_allclose(numpy_values, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(torch_values, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(torch_values, numpy_values, rtol=1e-6, atol=1e-6)
    for key in numpy_metrics:
        assert torch_metrics[key] == pytest.approx(numpy_metrics[key], rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("victim_policy_mode", ["greedy", "epsilon_greedy"])
@pytest.mark.parametrize("include_immediate", [False, True])
def test_numpy_torch_immediate_future_decomposition(
    victim_policy_mode: str,
    include_immediate: bool,
) -> None:
    numpy_values, numpy_metrics = _numpy_values(
        victim_policy_mode=victim_policy_mode,
        oracle_rollout_policy="greedy_best_response",
        include_immediate=include_immediate,
    )
    torch_values, torch_metrics = _torch_values(
        victim_policy_mode=victim_policy_mode,
        oracle_rollout_policy="greedy_best_response",
        include_immediate=include_immediate,
    )

    np.testing.assert_allclose(torch_values, numpy_values, rtol=1e-6, atol=1e-6)
    expected_mean = numpy_metrics["rollout_lola_future_profit"]
    if include_immediate:
        expected_mean += numpy_metrics["rollout_lola_first_step_profit"]
    assert float(np.mean(numpy_values)) == pytest.approx(expected_mean)
    assert float(np.mean(torch_values)) == pytest.approx(expected_mean, rel=1e-6, abs=1e-6)
    assert torch_metrics["rollout_lola_first_step_profit"] == pytest.approx(
        numpy_metrics["rollout_lola_first_step_profit"], rel=1e-6, abs=1e-6
    )
    assert torch_metrics["rollout_lola_future_profit"] == pytest.approx(
        numpy_metrics["rollout_lola_future_profit"], rel=1e-6, abs=1e-6
    )


def test_epsilon_greedy_deterministic_setup_matches_greedy() -> None:
    greedy_numpy, _ = _numpy_values(victim_policy_mode="greedy")
    epsilon_numpy, _ = _numpy_values(victim_policy_mode="epsilon_greedy")
    greedy_torch, _ = _torch_values(victim_policy_mode="greedy")
    epsilon_torch, _ = _torch_values(victim_policy_mode="epsilon_greedy")

    # exp(-beta * t) underflows to zero for the fixed beta/t, while still
    # exercising each backend's epsilon-greedy policy path.
    np.testing.assert_array_equal(epsilon_numpy, greedy_numpy)
    np.testing.assert_array_equal(epsilon_torch, greedy_torch)
    np.testing.assert_allclose(epsilon_torch, epsilon_numpy, rtol=1e-6, atol=1e-6)


def test_candidate_argmax_and_selected_actions_match() -> None:
    numpy_values, _ = _numpy_values(oracle_rollout_policy="greedy_best_response")
    torch_values, _ = _torch_values(oracle_rollout_policy="greedy_best_response")
    expected_actions = np.array([2, 2], dtype=np.int64)

    np.testing.assert_array_equal(np.argmax(numpy_values, axis=1), expected_actions)
    np.testing.assert_array_equal(np.argmax(torch_values, axis=1), expected_actions)

    numpy_actions, _ = tabular_rollout_lola_select_actions(
        numpy_values,
        tau=1e-6,
        epsilon=0.0,
        generator=torch.Generator(device="cpu").manual_seed(19),
        device=torch.device("cpu"),
        price_grid=PRICE_GRID,
    )
    torch_actions, _ = tabular_rollout_lola_select_actions(
        torch_values,
        tau=1e-6,
        epsilon=0.0,
        generator=torch.Generator(device="cpu").manual_seed(19),
        device=torch.device("cpu"),
        price_grid=PRICE_GRID,
    )
    np.testing.assert_array_equal(numpy_actions.cpu().numpy(), expected_actions)
    np.testing.assert_array_equal(torch_actions.cpu().numpy(), expected_actions)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_fixed_seed_reproduces_candidate_values_and_metrics(backend: str) -> None:
    run = _numpy_values if backend == "numpy" else _torch_values
    first_values, first_metrics = run(
        victim_policy_mode="epsilon_greedy",
        oracle_rollout_policy="greedy_best_response",
        seed=23,
    )
    second_values, second_metrics = run(
        victim_policy_mode="epsilon_greedy",
        oracle_rollout_policy="greedy_best_response",
        seed=23,
    )

    np.testing.assert_array_equal(first_values, second_values)
    assert first_metrics == second_metrics
