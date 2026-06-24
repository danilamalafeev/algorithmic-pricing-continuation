from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace

from scripts.run_pc_architecture_100k import CELLS, command_for, task_config, write_outputs


def runner_args(tmp_path) -> Namespace:
    return Namespace(
        root=str(tmp_path),
        results_root=str(tmp_path),
        seeds="0,1,2",
        cells=",".join(CELLS),
        total_steps=100_000,
        B=64,
        H=8,
        K=15,
        eval_every=5_000,
        eval_steps=2_000,
        log_every=1_000,
        batch_size=256,
        train_every=4,
        target_update_every=1_000,
        trajectory_diagnostic_steps=1_000,
        device="cuda",
        hypothesis_id="H-PC-TEST",
        force=False,
        dry_run=False,
    )


def test_default_matrix_excludes_plain_dqn_and_failed_classification_target(tmp_path):
    assert "dqn" not in CELLS
    assert "greedy_action_classification" not in {
        values.get("q_decoder_target") for values in CELLS.values()
    }

    args = runner_args(tmp_path)
    config = task_config(
        args,
        "qdecoder_centered_advantages",
        2,
        tmp_path / "qdecoder_centered_advantages" / "seed_2",
    )
    command = command_for(config, "qdecoder_centered_advantages")

    assert config["total_steps"] == 100_000
    assert config["save_final_state"] is True
    assert config["save_trajectory_diagnostics"] is True
    assert "--q-decoder-target" in command
    assert "centered_advantages" in command
    assert "--device" in command
    assert "cuda" in command


def test_manifest_documents_baseline_and_resume_limit(tmp_path):
    args = runner_args(tmp_path)
    out_dir = tmp_path / "victim_aware_dqn" / "seed_0"
    out_dir.mkdir(parents=True)
    (out_dir / "summary.json").write_text(
        json.dumps({"oracle_kind": "dqn_victim_aware", "seed": 0}),
        encoding="utf-8",
    )
    records = [
        {
            "cell": "victim_aware_dqn",
            "seed": 0,
            "out_dir": str(out_dir),
            "status": "success",
            "spec_id": "spec",
            "run_id": "run",
        }
    ]

    write_outputs(tmp_path, records, args, ["victim_aware_dqn"])

    manifest = json.loads((tmp_path / "study_manifest.json").read_text(encoding="utf-8"))
    assert "Not rerun" in manifest["notes"]["dqn_baseline"]
    assert "not implemented" in manifest["notes"]["victim_initialization"]
    assert "restart" in manifest["notes"]["resume"]


def test_cli_dry_run_plans_six_cells_times_three_seeds(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_pc_architecture_100k",
            "--root",
            str(tmp_path),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    commands = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(commands) == 18
    assert all("--total-steps 100000" in line for line in commands)
    assert all("--oracle-kind dqn " not in f"{line} " for line in commands)
    assert not (tmp_path / "study_manifest.json").exists()
