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
from scripts.analyze_mature_victim_matched_10m_1m import run as analyze_1m
from scripts.run_mature_victim_gate import task_config
from scripts.run_mature_victim_matched_10m_1m import (
    EXPECTED_10M_CHECKPOINT_SHA,
    runner_args,
)


REJECTED = (
    "shared_jepa",
    "qdecoder",
    "victim_aware",
    "variance",
    "imitation_bc_frozen",
    "rollout_lola",
)


def _write_checkpoint(path: Path, *, B: int = 64, K: int = 15) -> str:
    _, price_grid, _, profit_matrix = make_calvano_vec_env(B, H=8, K=K, seed=0)
    np.savez_compressed(
        path,
        kind=np.asarray("adaptive_q"),
        Q=np.zeros((B, K * K, K), dtype=np.float64),
        state_id=np.zeros(B, dtype=np.int64),
        t=np.full(B, 10_000_000, dtype=np.int64),
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


def test_1m_runner_dry_run_schedules_only_control_and_admitted_candidate(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mature_10m.npz"
    digest = _write_checkpoint(checkpoint)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_mature_victim_matched_10m_1m",
            "--victim-state",
            str(checkpoint),
            "--expected-checkpoint-sha",
            digest,
            "--root",
            str(tmp_path / "matched"),
            "--results-root",
            str(tmp_path),
            "--device",
            "cpu",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    commands = [line for line in completed.stdout.splitlines() if line.strip()]
    joined = "\n".join(commands)
    assert len(commands) == 6
    assert all("--total-steps 1000000" in command for command in commands)
    assert all("--checkpoint-every 25000" in command for command in commands)
    assert all("--eval-every 25000" in command for command in commands)
    assert sum("--oracle-kind dqn " in f"{command} " for command in commands) == 3
    assert sum("--oracle-kind imitation_option_dqn " in f"{command} " for command in commands) == 3
    assert not any(rejected in joined for rejected in REJECTED)


def test_1m_runner_pins_10m_sha_and_uses_atomic_resume_path(tmp_path: Path) -> None:
    assert EXPECTED_10M_CHECKPOINT_SHA == "cf47aa1806ac11bb65ddcbad82465e08cab97545181c2b3cb4cfb384afaad08d"
    checkpoint = tmp_path / "mature_10m.npz"
    args = runner_args(
        Namespace(
            victim_state=str(checkpoint),
            expected_checkpoint_sha=EXPECTED_10M_CHECKPOINT_SHA,
            root=str(tmp_path / "matched"),
            results_root=str(tmp_path),
            seeds="0,1,2",
            total_steps=1_000_000,
            eval_every=25_000,
            eval_steps=2_000,
            log_every=1_000,
            checkpoint_every=25_000,
            device="cpu",
            dry_run=True,
        )
    )
    assert args.expected_checkpoint_sha == EXPECTED_10M_CHECKPOINT_SHA
    assert args.cells == "dqn_control,imitation_option_dqn"
    assert args.auto_resume is True
    out_dir = tmp_path / "matched" / "dqn_control" / "seed_0"
    out_dir.mkdir(parents=True)
    checkpoint_path = out_dir / "training_checkpoint.pt"
    checkpoint_path.write_bytes(b"partial")
    config = task_config(args, "dqn_control", 0, out_dir, EXPECTED_10M_CHECKPOINT_SHA)
    assert config["resume_from"] == str(checkpoint_path)
    assert config["checkpoint_every"] == 25_000


def _metric_row(step: int, profit: float, price: float) -> dict[str, float | int]:
    row: dict[str, float | int] = {"step": step}
    for mode in ("continuation_adaptive", "fresh_adaptive", "continuation_frozen_greedy"):
        row[f"eval_{mode}_avg_profit_oracle"] = profit
        row[f"eval_{mode}_avg_profit_victim"] = profit - 0.01
        row[f"eval_{mode}_market_price_mean"] = price
        row[f"eval_{mode}_profit_asymmetry"] = 0.01
        for option in ("HOLD_HIGH", "HARVEST_UNDERCUT_1", "RESET_HIGH"):
            row[f"eval_{mode}_teacher_option_freq_{option}"] = 1.0 / 3.0
    return row


def _write_result(root: Path, cell: str, seed: int, sha: str, profits: dict[int, float]) -> dict[str, object]:
    out_dir = root / cell / f"seed_{seed}"
    out_dir.mkdir(parents=True)
    (out_dir / "config.json").write_text(
        json.dumps({"initial_victim_state_sha256": sha}),
        encoding="utf-8",
    )
    (out_dir / "initial_victim_state.json").write_text(
        json.dumps({"compatibility": "verified"}),
        encoding="utf-8",
    )
    rows = [_metric_row(step, profits[step], 1.72) for step in (100_000, 250_000, 500_000, 1_000_000)]
    pd.DataFrame(rows).to_csv(out_dir / "eval_metrics.csv", index=False)
    return {
        "cell": cell,
        "seed": seed,
        "out_dir": str(out_dir),
        "status": "success",
        "spec_id": f"spec-{cell}",
        "run_id": f"run-{cell}-{seed}",
    }


def _write_study(root: Path, sha: str) -> None:
    tasks = []
    for seed in (0, 1, 2):
        control = {100_000: 0.28, 250_000: 0.281, 500_000: 0.282, 1_000_000: 0.283 + seed * 0.001}
        candidate = {100_000: 0.30, 250_000: 0.305, 500_000: 0.308, 1_000_000: 0.313 + seed * 0.001}
        tasks.append(_write_result(root, "dqn_control", seed, sha, control))
        tasks.append(_write_result(root, "imitation_option_dqn", seed, sha, candidate))
    (root / "study_manifest.json").write_text(
        json.dumps(
            {
                "study_id": root.name,
                "hypothesis_id": "test",
                "victim_checkpoint_sha256": sha,
                "tasks": tasks,
            }
        ),
        encoding="utf-8",
    )


def test_1m_analyzer_computes_paired_deltas_and_admission(tmp_path: Path) -> None:
    root = tmp_path / "matched"
    output = tmp_path / "analysis"
    sha = "sha-test"
    _write_study(root, sha)
    analyze_1m(Namespace(root=str(root), output_dir=str(output), expected_checkpoint_sha=sha))
    summary = json.loads((output / "admission_summary.json").read_text(encoding="utf-8"))
    paired = pd.read_csv(output / "paired_checkpoint_metrics.csv")
    drift = pd.read_csv(output / "drift_by_interval.csv")
    final = paired[paired["step"] == 1_000_000]
    assert summary["decision"] == "STRONG_ADMIT"
    assert summary["winning_seed_count"] == 3
    assert summary["mean_paired_profit_delta"] == pytest.approx(0.03)
    assert final["continuation_adaptive_oracle_profit_delta_vs_dqn"].tolist() == pytest.approx([0.03, 0.03, 0.03])
    assert "continuation_adaptive_oracle_profit_delta_100k_to_1000k" in drift.columns


def test_1m_analyzer_rejects_mismatched_checkpoint_sha(tmp_path: Path) -> None:
    root = tmp_path / "matched"
    _write_study(root, "sha-in-manifest")
    with pytest.raises(RuntimeError, match="checkpoint SHA mismatch"):
        analyze_1m(
            Namespace(
                root=str(root),
                output_dir=str(tmp_path / "analysis"),
                expected_checkpoint_sha="different-sha",
            )
        )
