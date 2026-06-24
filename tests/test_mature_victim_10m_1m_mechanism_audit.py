from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_mature_victim_matched_10m_1m_mechanism import (
    eval_consistency,
    file_inventory,
    run,
    trajectory_mechanism,
)


def _metric_row(step: int, profit: float, market_price: float, harvest_freq: float = 0.0) -> dict[str, float | int]:
    row: dict[str, float | int] = {"step": step}
    for mode in ("continuation_adaptive", "fresh_adaptive", "continuation_frozen_greedy"):
        row[f"eval_{mode}_avg_profit_oracle"] = profit
        row[f"eval_{mode}_avg_profit_victim"] = profit - 0.02
        row[f"eval_{mode}_market_price_mean"] = market_price
        row[f"eval_{mode}_profit_asymmetry"] = 0.02
        row[f"eval_{mode}_teacher_option_freq_HOLD_HIGH"] = 0.0
        row[f"eval_{mode}_teacher_option_freq_HARVEST_UNDERCUT_1"] = harvest_freq
        row[f"eval_{mode}_teacher_option_freq_RESET_HIGH"] = 1.0 - harvest_freq
    return row


def _write_run(root: Path, cell: str, seed: int, profit: float, harvest_freq: float = 0.0) -> dict[str, object]:
    out_dir = root / cell / f"seed_{seed}"
    out_dir.mkdir(parents=True)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "oracle_kind": "imitation_option_dqn" if cell == "imitation_option_dqn" else "dqn",
                "total_steps": 1_000_000,
                "initial_victim_state_sha256": "sha",
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps({"seed": seed, "oracle_kind": cell}),
        encoding="utf-8",
    )
    (out_dir / "train_metrics.csv").write_text("step,loss\n1000000,0.0\n", encoding="utf-8")
    (out_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (out_dir / "progress.jsonl").write_text('{"step": 1000000}\n', encoding="utf-8")
    pd.DataFrame(
        [
            _metric_row(100_000, profit - 0.03, 1.61, harvest_freq),
            _metric_row(250_000, profit - 0.02, 1.60, harvest_freq),
            _metric_row(500_000, profit - 0.01, 1.60, harvest_freq),
            _metric_row(1_000_000, profit, 1.72, harvest_freq),
        ]
    ).to_csv(out_dir / "eval_metrics.csv", index=False)
    return {
        "cell": cell,
        "seed": seed,
        "status": "success",
        "spec_id": f"spec-{cell}",
        "run_id": f"run-{cell}-{seed}",
    }


def _write_study(root: Path) -> None:
    tasks = []
    tasks.append(_write_run(root, "dqn_control", 0, 0.27))
    tasks.append(_write_run(root, "imitation_option_dqn", 0, 0.324, 0.999))
    (root / "study_manifest.json").write_text(
        json.dumps({"victim_checkpoint_sha256": "sha", "tasks": tasks}),
        encoding="utf-8",
    )


def test_inventory_reports_missing_raw_bundle(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    inventory = file_inventory(root)
    missing_required = inventory[(inventory["required"]) & (~inventory["exists"])]
    assert len(missing_required) == 6 * 6 + 1
    assert "study_manifest.json" in missing_required["file"].tolist()


def test_eval_consistency_matches_analysis_profit(tmp_path: Path) -> None:
    root = tmp_path / "matched"
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    _write_study(root)
    pd.DataFrame(
        [
            {
                "seed": 0,
                "step": 1_000_000,
                "continuation_adaptive_oracle_profit": 0.324,
                "control_continuation_adaptive_oracle_profit": 0.27,
            }
        ]
    ).to_csv(analysis / "paired_checkpoint_metrics.csv", index=False)

    consistency = eval_consistency(root, analysis)
    candidate = consistency[consistency["cell"] == "imitation_option_dqn"].iloc[0]
    control = consistency[consistency["cell"] == "dqn_control"].iloc[0]

    assert candidate["eval_final_profit"] == pytest.approx(0.324)
    assert candidate["profit_drift_100k_to_1m"] == pytest.approx(0.03)
    assert candidate["eval_final_harvest_option_freq"] == pytest.approx(0.999)
    assert bool(candidate["matches_analysis_profit"]) is True
    assert bool(control["matches_analysis_profit"]) is True


def test_trajectory_mechanism_extracts_option_lock_in(tmp_path: Path) -> None:
    root = tmp_path / "matched"
    _write_study(root)
    traj_dir = root / "imitation_option_dqn" / "seed_0" / "trajectory_diagnostics"
    traj_dir.mkdir()
    pd.DataFrame(
        [
            {
                "oracle_option_name": "HARVEST_UNDERCUT_1",
                "oracle_action": 4,
                "victim_action": 5,
                "oracle_price": 1.70,
                "victim_price": 1.74,
                "market_price": 1.72,
                "oracle_profit": 0.33,
                "victim_profit": 0.29,
                "deviation_flag": True,
                "compliance_flag": False,
            }
            for _ in range(10)
        ]
    ).to_csv(traj_dir / "final_eval_continuation_adaptive.csv", index=False)

    mechanism, distributions = trajectory_mechanism(root)
    candidate = mechanism[mechanism["cell"] == "imitation_option_dqn"].iloc[0]

    assert bool(candidate["trajectory_exists"]) is True
    assert candidate["dominant_option"] == "HARVEST_UNDERCUT_1"
    assert candidate["dominant_option_freq"] == pytest.approx(1.0)
    assert candidate["price_gap_victim_minus_oracle_mean"] == pytest.approx(0.04)
    assert not distributions.empty


def test_audit_run_writes_report(tmp_path: Path) -> None:
    root = tmp_path / "matched"
    output = tmp_path / "audit"
    _write_study(root)

    run(Namespace(root=str(root), analysis_dir="", output_dir=str(output)))

    report = (output / "RAW_AUDIT_REPORT.md").read_text(encoding="utf-8")
    assert "Required raw bundle complete: True" in report
    assert (output / "eval_consistency.csv").exists()
