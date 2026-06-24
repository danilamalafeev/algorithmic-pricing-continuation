from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace

import pandas as pd

from scripts.evaluate_harvest_imitation_gate import evaluate
from scripts.run_harvest_imitation_gate import (
    CELLS,
    command_for,
    parse_overrides,
    task_config,
    write_outputs,
)


def runner_args(tmp_path) -> Namespace:
    return Namespace(
        root=str(tmp_path),
        results_root=str(tmp_path),
        seeds="0,1",
        cells=",".join(CELLS),
        total_steps=5_000,
        B=16,
        H=8,
        K=15,
        eval_every=2_500,
        eval_steps=1_000,
        log_every=500,
        batch_size=64,
        train_every=2,
        target_update_every=500,
        device="cpu",
        imitation_config=[],
        hypothesis_id="H-IMITATION-TEST",
        force=False,
        dry_run=False,
    )


def test_default_cells_and_future_imitation_overrides(tmp_path):
    args = runner_args(tmp_path)
    overrides = parse_overrides(
        ["imitation_bc_steps=750", "imitation_freeze_encoder=true", "imitation_dataset=teacher"]
    )
    config = task_config(
        args,
        "imitation_option_dqn",
        1,
        tmp_path / "imitation_option_dqn" / "seed_1",
        overrides,
    )
    command = command_for(config, overrides)

    assert list(CELLS) == [
        "dqn",
        "dqn_victim_aware",
        "scripted_harvest_undercut",
        "imitation_bc_frozen",
        "imitation_option_dqn",
    ]
    assert config["total_steps"] == 5_000
    assert config["imitation_bc_steps"] == 750
    assert "--imitation-bc-steps" in command
    assert "--imitation-freeze-encoder" in command
    assert "--imitation-dataset" in command
    assert "--eval-modes" in command
    assert any("continuation_adaptive" in value for value in command)


def _write_result(path, cell: str, seed: int, profit: float) -> None:
    path.mkdir(parents=True)
    summary = {
        "oracle_kind": cell,
        "seed": seed,
        "final_eval_continuation_adaptive_avg_profit_oracle": profit,
        "final_eval_continuation_adaptive_market_price_mean": 1.82,
        "bc_validation_loss": 0.2,
        "bc_validation_accuracy": 0.8,
        "final_eval_continuation_adaptive_phase_freq_HOLD": 0.55,
        "final_eval_continuation_adaptive_phase_freq_HARVEST": 0.45,
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_write_outputs_creates_manifest_seed_and_spec_aggregates(tmp_path):
    args = runner_args(tmp_path)
    records = []
    for cell, profit in (("dqn", 0.30), ("imitation_option_dqn", 0.35)):
        for seed in (0, 1):
            out_dir = tmp_path / cell / f"seed_{seed}"
            _write_result(out_dir, cell, seed, profit + 0.01 * seed)
            records.append(
                {
                    "cell": cell,
                    "seed": seed,
                    "out_dir": str(out_dir),
                    "status": "success",
                    "spec_id": f"spec-{cell}",
                    "run_id": f"run-{cell}-{seed}",
                }
            )

    write_outputs(tmp_path, records, args)

    summary = pd.read_csv(tmp_path / "summary_by_seed.csv")
    aggregate = pd.read_csv(tmp_path / "aggregate_by_spec.csv")
    manifest = json.loads((tmp_path / "study_manifest.json").read_text(encoding="utf-8"))
    assert len(summary) == 4
    assert set(aggregate["cell"]) == {"dqn", "imitation_option_dqn"}
    assert set(aggregate["completed_seeds"]) == {2}
    assert manifest["cells"] == list(CELLS)
    assert len(manifest["tasks"]) == 4


def _write_gate_summary(root, *, imitation_profits=(0.35, 0.36), market=1.82, include_bc=True):
    rows = []
    for seed, profit in enumerate((0.30, 0.32)):
        rows.append(
            {
                "cell": "dqn",
                "seed": seed,
                "final_eval_continuation_adaptive_avg_profit_oracle": profit,
                "final_eval_continuation_adaptive_market_price_mean": 1.80,
            }
        )
    for seed, profit in enumerate(imitation_profits):
        rows.append(
            {
                "cell": "imitation_bc_frozen",
                "seed": seed,
                "final_eval_continuation_adaptive_avg_profit_oracle": profit + 0.002,
                "final_eval_continuation_adaptive_market_price_mean": market,
                "final_eval_continuation_adaptive_profit_asymmetry": 0.04,
            }
        )
        row = {
            "cell": "imitation_option_dqn",
            "seed": seed,
            "final_eval_continuation_adaptive_avg_profit_oracle": profit,
            "final_eval_continuation_adaptive_market_price_mean": market,
            "final_eval_continuation_adaptive_profit_asymmetry": 0.05,
            "final_eval_continuation_adaptive_teacher_option_freq_HOLD_HIGH": 0.60,
            "final_eval_continuation_adaptive_teacher_option_freq_HARVEST_UNDERCUT_1": 0.40,
        }
        if include_bc:
            row["bc_validation_loss"] = 0.2
            row["bc_validation_accuracy"] = 0.8
            row["bc_validation_majority_accuracy"] = 0.6
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "summary_by_seed.csv", index=False)


def test_evaluator_admits_complete_stable_signal(tmp_path):
    _write_gate_summary(tmp_path)
    output = tmp_path / "analysis"

    details = evaluate(tmp_path, output)
    decision = json.loads((output / "admission_summary.json").read_text(encoding="utf-8"))

    assert len(details) == 2
    assert decision["imitation_profit_beats_dqn"] is True
    assert decision["no_market_price_collapse"] is True
    assert decision["positive_profit_asymmetry"] is True
    assert decision["bc_frozen_high_price_positive_asymmetry"] is True
    assert decision["rl_retains_bc_profit"] is True
    assert decision["rl_retains_bc_market_price"] is True
    assert decision["stable_two_seeds"] is True
    assert decision["phase_frequencies_nondegenerate"] is True
    assert decision["bc_validation_metrics_available"] is True
    assert decision["bc_validation_beats_majority"] is True
    assert decision["decision"] == "ADMIT"


def test_evaluator_rejects_collapse_instability_and_missing_bc(tmp_path):
    _write_gate_summary(
        tmp_path,
        imitation_profits=(0.29, 0.40),
        market=1.60,
        include_bc=False,
    )
    output = tmp_path / "analysis"

    evaluate(tmp_path, output)
    decision = json.loads((output / "admission_summary.json").read_text(encoding="utf-8"))

    assert decision["no_market_price_collapse"] is False
    assert decision["stable_two_seeds"] is False
    assert decision["bc_validation_metrics_available"] is False
    assert decision["decision"] == "REJECT"


def test_runner_cli_dry_run_plans_ten_5k_tasks_without_outputs(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_harvest_imitation_gate",
            "--root",
            str(tmp_path),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    commands = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(commands) == 10
    assert all("--total-steps 5000" in line for line in commands)
    assert any("--oracle-kind imitation_option_dqn" in line for line in commands)
    assert not (tmp_path / "study_manifest.json").exists()
