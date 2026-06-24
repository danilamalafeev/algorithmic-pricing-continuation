from __future__ import annotations

import subprocess
import sys
from argparse import Namespace

from scripts.run_mature_victim_gate import task_config


def test_1m_runner_dry_run_is_matched_and_resumable(tmp_path):
    checkpoint = (
        "results/mature_q_vs_q_victim/seed_0/mature_victim_state.npz"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_mature_victim_matched_1m",
            "--victim-state",
            checkpoint,
            "--root",
            str(tmp_path / "study"),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    commands = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(commands) == 6
    assert all("--total-steps 1000000" in command for command in commands)
    assert all("--checkpoint-every 10000" in command for command in commands)
    assert sum("--oracle-kind dqn " in f"{command} " for command in commands) == 3
    assert sum("--oracle-kind imitation_option_dqn " in f"{command} " for command in commands) == 3
    assert all("--force" not in command for command in commands)


def test_partial_1m_task_uses_existing_training_checkpoint(tmp_path):
    out_dir = tmp_path / "dqn_control" / "seed_0"
    out_dir.mkdir(parents=True)
    checkpoint = out_dir / "training_checkpoint.pt"
    checkpoint.write_bytes(b"partial")
    args = Namespace(
        victim_state="mature.npz",
        root=str(tmp_path),
        total_steps=1_000_000,
        B=64,
        H=8,
        K=15,
        eval_every=10_000,
        eval_steps=2_000,
        log_every=1_000,
        checkpoint_every=10_000,
        auto_resume=True,
        batch_size=256,
        train_every=4,
        target_update_every=1_000,
        trajectory_diagnostic_steps=1_000,
        device="cuda",
        hypothesis_id="H-MATURE-VICTIM-1M-001",
    )

    config = task_config(args, "dqn_control", 0, out_dir, "sha")

    assert config["resume_from"] == str(checkpoint)
