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
from scripts.analyze_mature_victim_3m_1m_followup import run as analyze_3m
from scripts.run_mature_victim_3m_1m_followup import EXPECTED_3M_CHECKPOINT_SHA, runner_args
from scripts.run_mature_victim_gate import task_config


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
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        kind=np.asarray("adaptive_q"),
        Q=np.zeros((B, K * K, K), dtype=np.float64),
        state_id=np.zeros(B, dtype=np.int64),
        t=np.full(B, 3_000_000, dtype=np.int64),
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


def test_3m_followup_runner_dry_run_uses_only_matched_cells_and_pins_sha(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mature_3m.npz"
    digest = _write_checkpoint(checkpoint)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_mature_victim_3m_1m_followup",
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
    assert digest in joined
    assert not any(marker in joined for marker in REJECTED)


def test_3m_followup_runner_configures_auto_resume_and_expected_sha(tmp_path: Path) -> None:
    assert EXPECTED_3M_CHECKPOINT_SHA == "2845e31c4d37232c0a9fc4e2314b601169776e91e44b04b3402ae9f4d4f3c867"
    args = runner_args(
        Namespace(
            victim_state=str(tmp_path / "mature_3m.npz"),
            expected_checkpoint_sha=EXPECTED_3M_CHECKPOINT_SHA,
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
    assert args.cells == "dqn_control,imitation_option_dqn"
    assert args.auto_resume is True
    assert args.expected_checkpoint_sha == EXPECTED_3M_CHECKPOINT_SHA
    out_dir = tmp_path / "matched" / "dqn_control" / "seed_0"
    out_dir.mkdir(parents=True)
    checkpoint_path = out_dir / "training_checkpoint.pt"
    checkpoint_path.write_bytes(b"partial")
    config = task_config(args, "dqn_control", 0, out_dir, EXPECTED_3M_CHECKPOINT_SHA)
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


def _write_result(root: Path, cell: str, seed: int, sha: str, profits: dict[int, float], price: float) -> dict[str, object]:
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
    rows = [_metric_row(step, profits[step], price) for step in (100_000, 250_000, 500_000, 1_000_000)]
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
                "market_price": [1.71, 1.71, 1.77],
            }
        ).to_csv(diag / "final_eval_continuation_adaptive.csv", index=False)
    return {
        "cell": cell,
        "seed": seed,
        "out_dir": str(out_dir),
        "status": "success",
        "spec_id": f"spec-{cell}",
        "run_id": f"run-{cell}-{seed}",
    }


def _write_study(root: Path, *, candidate_final: float, control_final: float, price: float, sha: str = "sha-3m") -> None:
    tasks = []
    for seed in (0, 1, 2):
        control = {100_000: control_final - 0.01, 250_000: control_final - 0.005, 500_000: control_final, 1_000_000: control_final}
        candidate = {100_000: candidate_final - 0.02, 250_000: candidate_final - 0.01, 500_000: candidate_final - 0.005, 1_000_000: candidate_final}
        tasks.append(_write_result(root, "dqn_control", seed, sha, control, price))
        tasks.append(_write_result(root, "imitation_option_dqn", seed, sha, candidate, price))
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


def test_3m_analyzer_classifies_collusion_level_and_reads_mechanism(tmp_path: Path) -> None:
    root = tmp_path / "matched"
    output = tmp_path / "analysis"
    _write_study(root, candidate_final=0.325, control_final=0.28, price=1.72)
    decision = analyze_3m(Namespace(root=str(root), output_dir=str(output), expected_checkpoint_sha="sha-3m"))
    paired = pd.read_csv(output / "paired_checkpoint_metrics.csv")
    mechanism = pd.read_csv(output / "mechanism_summary.csv").iloc[0]
    assert decision["decision"] == "COLLUSION_LEVEL"
    assert decision["mean_paired_profit_delta"] == pytest.approx(0.045)
    assert paired[paired["step"] == 1_000_000]["continuation_adaptive_oracle_profit_delta_vs_dqn"].tolist() == pytest.approx([0.045, 0.045, 0.045])
    assert mechanism["dominant_option"] == "HARVEST_UNDERCUT_1"
    assert int(mechanism["dominant_oracle_action"]) == 7
    assert int(mechanism["dominant_victim_action"]) == 8


def test_3m_analyzer_classifies_below_collusion_but_stable(tmp_path: Path) -> None:
    root = tmp_path / "matched"
    _write_study(root, candidate_final=0.31, control_final=0.28, price=1.70)
    decision = analyze_3m(Namespace(root=str(root), output_dir=str(tmp_path / "analysis"), expected_checkpoint_sha="sha-3m"))
    assert decision["decision"] == "BELOW_COLLUSION_BUT_STABLE"
    assert decision["strong_3m_admit"] is True
    assert decision["collusion_level"] is False


def test_3m_analyzer_classifies_reject_investigate(tmp_path: Path) -> None:
    root = tmp_path / "matched"
    _write_study(root, candidate_final=0.27, control_final=0.28, price=1.55)
    decision = analyze_3m(Namespace(root=str(root), output_dir=str(tmp_path / "analysis"), expected_checkpoint_sha="sha-3m"))
    assert decision["decision"] == "REJECT_INVESTIGATE"
    assert decision["strong_3m_admit"] is False
