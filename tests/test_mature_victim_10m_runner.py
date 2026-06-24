from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

from experiments.dqn_oracle_config import make_calvano_vec_env
from experiments.dqn_oracle_tabular import victim_market_fingerprint, victim_state_sha256
from experiments.experiment_registry import find_completed_run
from scripts.run_mature_victim_gate import task_config


REJECTED = (
    "shared_jepa",
    "qdecoder",
    "victim_aware",
    "variance",
    "imitation_bc_frozen",
)


def _write_checkpoint(path: Path, *, B: int = 4, K: int = 5) -> str:
    _, price_grid, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
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


def _runner_namespace(tmp_path: Path, checkpoint: Path) -> Namespace:
    return Namespace(
        victim_state=str(checkpoint),
        root=str(tmp_path / "matched"),
        results_root=str(tmp_path),
        cells="dqn_control,imitation_option_dqn",
        seeds="0,1,2",
        total_steps=100_000,
        B=4,
        H=4,
        K=5,
        eval_every=10_000,
        eval_steps=200,
        log_every=1_000,
        checkpoint_every=10_000,
        auto_resume=True,
        batch_size=32,
        train_every=4,
        target_update_every=1_000,
        trajectory_diagnostic_steps=200,
        device="cpu",
        hypothesis_id="H-MATURE-VICTIM-10M-TEST",
        force=False,
        dry_run=False,
    )


def test_10m_matched_runner_dry_run_uses_only_admitted_cells(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mature_10m.npz"
    _write_checkpoint(checkpoint, B=64, K=15)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_mature_victim_matched_10m",
            "--victim-state",
            str(checkpoint),
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
    assert len(commands) == 6
    assert all("--total-steps 100000" in command for command in commands)
    assert sum("--oracle-kind dqn " in f"{command} " for command in commands) == 3
    assert sum("--oracle-kind imitation_option_dqn " in f"{command} " for command in commands) == 3
    assert not any(rejected in "\n".join(commands) for rejected in REJECTED)


def test_10m_runner_config_uses_auto_resume_and_registry_duplicate_detection(tmp_path: Path) -> None:
    checkpoint = tmp_path / "mature_10m.npz"
    digest = _write_checkpoint(checkpoint)
    args = _runner_namespace(tmp_path, checkpoint)
    out_dir = tmp_path / "matched" / "dqn_control" / "seed_0"
    out_dir.mkdir(parents=True)
    (out_dir / "training_checkpoint.pt").write_bytes(b"partial")

    config = task_config(args, "dqn_control", 0, out_dir, digest)
    assert config["oracle_kind"] == "dqn"
    assert config["resume_from"] == str(out_dir / "training_checkpoint.pt")

    existing = tmp_path / "old" / "dqn_control" / "seed_0"
    existing.mkdir(parents=True)
    (existing / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (existing / "summary.json").write_text("{}", encoding="utf-8")
    found = find_completed_run({**config, "out_dir": str(tmp_path / "elsewhere")}, tmp_path)
    assert found is not None
    assert found.path == existing
