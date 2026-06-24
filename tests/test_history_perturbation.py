from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.dqn_oracle_config import QVictimOracleConfig, make_calvano_vec_env
from experiments.dqn_oracle_evaluation import evaluate
from scripts.analyze_history_perturbation import (
    discover_runs,
    paired_metrics,
    parse_shock,
    recovery_step,
)


def test_parse_shock_manifest_fields() -> None:
    shock = parse_shock("oracle_random_5:oracle:random:5", start_step=7, default_seed=123)

    assert shock == {
        "id": "oracle_random_5",
        "target": "oracle",
        "action": "random",
        "start_step": 7,
        "duration": 5,
        "seed": 123,
    }


def test_recovery_step_detects_first_stable_window() -> None:
    baseline = pd.DataFrame({"step": range(8), "market_price": [2.0] * 8})
    shock = pd.DataFrame(
        {
            "step": range(8),
            "market_price": [2.0, 2.0, 1.0, 1.2, 1.7, 1.98, 2.0, 2.0],
        }
    )

    recovered = recovery_step(
        baseline,
        shock,
        metric="market_price",
        start_step=2,
        duration=1,
        tolerance=0.05,
        stable_steps=2,
    )

    assert recovered == 6.0


def test_paired_metrics_compute_imitation_dqn_deltas() -> None:
    metrics = pd.DataFrame(
        [
            {
                "root": "root",
                "seed": 0,
                "shock_id": "s",
                "cell": "dqn_control",
                "shock_oracle_profit": 0.3,
                "shock_market_price": 1.8,
                "oracle_profit_delta_vs_unshocked": -0.02,
            },
            {
                "root": "root",
                "seed": 0,
                "shock_id": "s",
                "cell": "imitation_option_dqn",
                "shock_oracle_profit": 0.34,
                "shock_market_price": 1.85,
                "oracle_profit_delta_vs_unshocked": -0.01,
            },
        ]
    )

    paired = paired_metrics(metrics)

    assert len(paired) == 1
    assert np.isclose(paired.loc[0, "imitation_minus_dqn_oracle_profit"], 0.04)
    assert np.isclose(paired.loc[0, "relative_shock_resilience"], 0.01)


def test_discover_runs_skips_missing_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "dqn_control" / "seed_0"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"seed": 0}), encoding="utf-8")

    runs, skipped = discover_runs([tmp_path], cells={"dqn_control"}, seeds=None)

    assert runs == []
    assert skipped[0]["reason"] == "missing_training_checkpoint"


def test_evaluate_action_perturbation_overrides_only_window() -> None:
    config = QVictimOracleConfig(
        oracle_kind="scripted_hold_high",
        B=3,
        H=2,
        K=5,
        seed=11,
        eval_steps=4,
        teacher_high_anchor="monopoly",
        device="cpu",
    )
    _, _, benchmarks, _ = make_calvano_vec_env(config.B, config.H, config.K, config.seed + 10_000)
    rows: list[dict] = []

    evaluate(
        config,
        params={},
        buffers={},
        benchmarks=benchmarks,
        trajectory_rows=rows,
        trajectory_limit_steps=4,
        perturbation={
            "id": "both_low_2",
            "target": "both",
            "action": "low",
            "start_step": 1,
            "duration": 2,
        },
    )

    df = pd.DataFrame(rows)
    active = df[df["perturbation_active"]]
    inactive = df[~df["perturbation_active"]]
    assert set(active["step"]) == {1, 2}
    assert set(active["oracle_action"]) == {0}
    assert set(active["victim_action"]) == {0}
    assert set(inactive["step"]) == {0, 3}
    assert active["unperturbed_oracle_action"].notna().all()
