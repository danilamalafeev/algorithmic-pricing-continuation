from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace

import numpy as np

from calvano_market import CalvanoMarketConfig, build_static_benchmarks
from calvano_qlearning import SessionResult
from experiments.dqn_oracle_config import make_calvano_vec_env
from experiments.dqn_oracle_tabular import (
    victim_market_fingerprint,
    victim_state_sha256,
)
from scripts.build_mature_q_vs_q_victim import victim_state_from_result
from scripts.run_mature_victim_gate import CELLS, command_for, task_config


def _write_checkpoint(path, *, B=4, K=5):
    _, price_grid, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    np.savez_compressed(
        path,
        kind=np.asarray("adaptive_q"),
        Q=np.zeros((B, K * K, K), dtype=np.float64),
        state_id=np.zeros(B, dtype=np.int64),
        t=np.full(B, 1_000_000, dtype=np.int64),
    )
    digest = victim_state_sha256(path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "sha256": digest,
                "victim_alpha": 0.15,
                "victim_beta": 4e-6,
                "victim_delta": 0.95,
                "market_fingerprint": victim_market_fingerprint(
                    price_grid,
                    profit_matrix,
                    K=K,
                ),
            }
        ),
        encoding="utf-8",
    )
    return digest


def _args(tmp_path, checkpoint) -> Namespace:
    return Namespace(
        victim_state=str(checkpoint),
        root=str(tmp_path / "gate"),
        results_root=str(tmp_path),
        cells=",".join(CELLS),
        seeds="0,1,2",
        total_steps=20_000,
        B=4,
        H=4,
        K=5,
        eval_every=2_500,
        eval_steps=200,
        log_every=500,
        batch_size=32,
        train_every=4,
        target_update_every=1_000,
        trajectory_diagnostic_steps=200,
        device="cpu",
        hypothesis_id="H-MATURE-TEST",
        force=False,
        dry_run=False,
    )


def test_q_vs_q_result_converts_to_replicated_victim_state():
    K = 5
    market = CalvanoMarketConfig(m=K)
    benchmarks = build_static_benchmarks(market)
    q = np.arange(2 * K * K * K, dtype=np.float64).reshape(2, K * K, K)
    result = SessionResult(
        converged=True,
        periods_to_convergence=123_456,
        final_greedy_policy=np.zeros((2, K * K), dtype=np.int64),
        final_q=q,
        last_prices=benchmarks.price_grid[[2, 3]],
        detected_cycle_length=1,
        long_run_avg_price=np.array([1.7, 1.8]),
        long_run_avg_profit=np.array([0.3, 0.31]),
        profit_gain_delta=np.array([0.5, 0.6]),
    )

    state = victim_state_from_result(
        result,
        price_grid=benchmarks.price_grid,
        B=4,
        K=K,
    )

    assert state["Q"].shape == (4, K * K, K)
    np.testing.assert_array_equal(state["Q"][0], q[1])
    assert np.all(state["state_id"] == 2 * K + 3)
    assert np.all(state["t"] == 123_456)


def test_gate_config_uses_checkpoint_hash_and_matched_control(tmp_path):
    checkpoint = tmp_path / "mature.npz"
    digest = _write_checkpoint(checkpoint)
    args = _args(tmp_path, checkpoint)
    config = task_config(
        args,
        "dqn_control",
        1,
        tmp_path / "gate" / "dqn_control" / "seed_1",
        digest,
    )
    command = command_for(config, "dqn_control")

    assert config["oracle_kind"] == "dqn"
    assert config["initial_victim_state_mode"] == "mature_initialization"
    assert config["initial_victim_state_sha256"] == digest
    assert "--initial-victim-state-sha256" in command
    assert "--initial-victim-state-mode" in command
    assert "--checkpoint-every 0" in " ".join(command)


def test_gate_cli_dry_run_plans_six_cells_times_three_seeds(tmp_path):
    checkpoint = tmp_path / "mature.npz"
    _write_checkpoint(checkpoint)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_mature_victim_gate",
            "--victim-state",
            str(checkpoint),
            "--root",
            str(tmp_path / "gate"),
            "--B",
            "4",
            "--H",
            "4",
            "--K",
            "5",
            "--device",
            "cpu",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    commands = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(commands) == 18
    assert all("--total-steps 20000" in line for line in commands)
    assert sum("--oracle-kind dqn " in f"{line} " for line in commands) == 3
    assert not (tmp_path / "gate" / "study_manifest.json").exists()
