from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.dqn_oracle_config import make_calvano_vec_env
from experiments.dqn_oracle_tabular import victim_market_fingerprint, victim_state_sha256
from scripts.analyze_mature_victim_maturity_sweep import (
    classify_maturity_pattern,
    run as analyze_sweep,
)
from scripts.run_mature_victim_maturity_sweep import AGE_GRID


REJECTED = (
    "shared_jepa",
    "qdecoder",
    "victim_aware",
    "variance",
    "imitation_bc_frozen",
    "rollout_lola",
)


def _write_checkpoint(path: Path, *, B: int = 4, H: int = 4, K: int = 5, steps: int = 0) -> str:
    _, price_grid, _, profit_matrix = make_calvano_vec_env(B, H=H, K=K, seed=0)
    np.savez_compressed(
        path,
        kind=np.asarray("adaptive_q"),
        Q=np.zeros((B, K * K, K), dtype=np.float64),
        state_id=np.zeros(B, dtype=np.int64),
        t=np.full(B, steps, dtype=np.int64),
    )
    digest = victim_state_sha256(path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "sha256": digest,
                "victim_alpha": 0.15,
                "victim_beta": 4e-6,
                "victim_delta": 0.95,
                "market_fingerprint": victim_market_fingerprint(price_grid, profit_matrix, K=K),
            }
        ),
        encoding="utf-8",
    )
    return digest


def test_maturity_sweep_dry_run_schedules_exact_age_grid_and_cells(tmp_path: Path) -> None:
    checkpoints = {}
    for age in AGE_GRID:
        path = tmp_path / "checkpoints" / age / "mature_victim_state.npz"
        path.parent.mkdir(parents=True)
        checkpoints[age] = (path, _write_checkpoint(path, steps=0 if age == "fresh" else 100_000))
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_mature_victim_maturity_sweep",
            "--root",
            str(tmp_path / "results"),
            "--results-root",
            str(tmp_path),
            "--fresh-state",
            str(checkpoints["fresh"][0]),
            "--q100k-state",
            str(checkpoints["100k"][0]),
            "--q1m-state",
            str(checkpoints["1m"][0]),
            "--q3m-state",
            str(checkpoints["3m"][0]),
            "--q10m-state",
            str(checkpoints["10m"][0]),
            "--expected-10m-sha",
            checkpoints["10m"][1],
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
    output = completed.stdout
    commands = [line for line in output.splitlines() if "experiments.dqn_oracle_vs_qvictim" in line]
    assert "[maturity-sweep] total_tasks=30" in output
    assert len(commands) == 30
    assert all("--total-steps 100000" in command for command in commands)
    assert sorted({part.split()[0] for part in []}) == []
    for age in AGE_GRID:
        assert output.count(f"age={age} ") == 6
    assert sum("--oracle-kind dqn " in f"{command} " for command in commands) == 15
    assert sum("--oracle-kind imitation_option_dqn " in f"{command} " for command in commands) == 15
    assert "would build 10m" not in output.lower()
    assert not any(marker in "\n".join(commands).lower() for marker in REJECTED)


def _metric_row(step: int, profit: float, price: float) -> dict[str, float | int]:
    return {
        "step": step,
        "eval_continuation_adaptive_avg_profit_oracle": profit,
        "eval_continuation_adaptive_avg_profit_victim": profit - 0.01,
        "eval_continuation_adaptive_market_price_mean": price,
        "eval_continuation_adaptive_profit_asymmetry": 0.01,
    }


def _write_result(root: Path, age: str, cell: str, seed: int, profit: float, price: float) -> dict[str, object]:
    out_dir = root / age / cell / f"seed_{seed}"
    out_dir.mkdir(parents=True)
    rows = [_metric_row(50_000, profit - 0.01, price), _metric_row(100_000, profit, price)]
    pd.DataFrame(rows).to_csv(out_dir / "eval_metrics.csv", index=False)
    if cell == "imitation_option_dqn":
        diag = out_dir / "trajectory_diagnostics"
        diag.mkdir()
        pd.DataFrame(
            {
                "oracle_option_name": ["HARVEST_UNDERCUT_1", "HARVEST_UNDERCUT_1", "HOLD_HIGH"],
                "oracle_action": [7, 7, 10],
                "victim_action": [8, 8, 8],
                "oracle_price": [1.69, 1.69, 1.81],
                "victim_price": [1.73, 1.73, 1.73],
                "undercut_ticks": [1, 1, np.nan],
            }
        ).to_csv(diag / "final_eval_continuation_adaptive.csv", index=False)
    return {
        "age": age,
        "cell": cell,
        "seed": seed,
        "out_dir": str(out_dir),
        "checkpoint_sha256": f"sha-{age}",
        "status": "success",
        "spec_id": f"spec-{age}-{cell}",
        "run_id": f"run-{age}-{cell}-{seed}",
    }


def _write_study(root: Path) -> None:
    tasks = []
    for age in AGE_GRID:
        for seed in (0, 1, 2):
            control_profit = 0.25
            candidate_profit = 0.31 if age != "10m" else 0.33
            tasks.append(_write_result(root, age, "dqn_control", seed, control_profit, 1.62))
            tasks.append(_write_result(root, age, "imitation_option_dqn", seed, candidate_profit, 1.72))
    (root / "study_manifest.json").write_text(
        json.dumps(
            {
                "study_id": "toy",
                "age_grid": list(AGE_GRID),
                "seeds": [0, 1, 2],
                "age_checkpoints": {age: {"sha256": f"sha-{age}"} for age in AGE_GRID},
                "tasks": tasks,
            }
        ),
        encoding="utf-8",
    )


def test_maturity_sweep_analyzer_paired_deltas_and_mechanism(tmp_path: Path) -> None:
    root = tmp_path / "results"
    output = tmp_path / "analysis"
    _write_study(root)
    summary = analyze_sweep(Namespace(root=str(root), output_dir=str(output)))
    paired = pd.read_csv(output / "paired_by_age_seed.csv")
    admission = pd.read_csv(output / "admission_by_age.csv")
    mechanism = pd.read_csv(output / "mechanism_by_age.csv")
    fresh = admission[admission["age"] == "fresh"].iloc[0]
    assert fresh["paired_delta"] == pytest.approx(0.06)
    assert int(fresh["wins"]) == 3
    ten_m = mechanism[mechanism["age"] == "10m"].iloc[0]
    assert ten_m["dominant_option"] == "HARVEST_UNDERCUT_1"
    assert int(ten_m["dominant_oracle_action"]) == 7
    assert int(ten_m["dominant_victim_action"]) == 8
    assert summary["classification"]["interpretation"] == "CREATION"
    assert len(paired) == 15


def test_maturity_sweep_classifies_creation_stabilization_and_mature_exploitation() -> None:
    base = pd.DataFrame(
        {
            "age": list(AGE_GRID),
            "strong_age_success": [False, False, False, False, False],
        }
    )
    creation = base.copy()
    creation.loc[creation["age"] == "fresh", "strong_age_success"] = True
    assert classify_maturity_pattern(creation)["interpretation"] == "CREATION"

    stabilization = base.copy()
    stabilization.loc[stabilization["age"] == "1m", "strong_age_success"] = True
    assert classify_maturity_pattern(stabilization)["interpretation"] == "EARLY_STABILIZATION"

    mature = base.copy()
    mature.loc[mature["age"] == "10m", "strong_age_success"] = True
    mechanism = pd.DataFrame(
        {
            "age": ["10m"],
            "dominant_option": ["HARVEST_UNDERCUT_1"],
            "dominant_option_frequency": [0.95],
            "dominant_oracle_action": [7],
        }
    )
    classified = classify_maturity_pattern(mature, mechanism)
    assert classified["interpretation"] == "MATURE_EXPLOITATION"
    assert classified["deterministic_10m_harvest_undercut"] is True
