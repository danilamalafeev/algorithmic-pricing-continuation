from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from scripts.analyze_mature_q_vs_q_checkpoint_10m import run as analyze_checkpoint
from scripts.run_mature_q_vs_q_checkpoint_10m import run as run_checkpoint


def _args(root: Path, total_steps: int) -> Namespace:
    return Namespace(
        root=str(root),
        total_steps=total_steps,
        checkpoint_every=100,
        report_every=100,
        chunk_steps=100,
        eval_periods=20,
        convergence_window=1_000,
        seed=0,
        B=4,
        K=5,
        alpha=0.15,
        beta=4e-6,
        delta=0.95,
        device="cpu",
        resume=True,
        stop_on_convergence=False,
    )


def test_q_vs_q_checkpoint_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    full_dir = tmp_path / "full"
    split_dir = tmp_path / "split"

    full = run_checkpoint(_args(full_dir, 200))
    run_checkpoint(_args(split_dir, 100))
    resumed = run_checkpoint(_args(split_dir, 200))

    assert resumed["resume_count"] == 1
    assert resumed["q_table_sha256"] == full["q_table_sha256"]
    assert resumed["checkpoint_sha256"] == full["checkpoint_sha256"]
    with np.load(full_dir / "mature_victim_state.npz") as left, np.load(split_dir / "mature_victim_state.npz") as right:
        assert left.files == right.files
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key])


def test_q_vs_q_checkpoint_summary_and_analyzer_schema(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    summary = run_checkpoint(_args(root, 100))
    expected = {
        "total_steps",
        "completed_steps",
        "converged",
        "final_symmetric_profit",
        "final_market_price",
        "checkpoint_sha256",
        "checkpoint_path",
        "resume_count",
        "completion_status",
        "q_table_sha256",
    }
    assert expected <= set(summary)
    assert summary["completed_steps"] == 100
    assert summary["completion_status"] == "complete"

    output_dir = tmp_path / "analysis"
    analyzed = analyze_checkpoint(Namespace(root=str(root), output_dir=str(output_dir)))
    assert analyzed["checkpoint_sha256"] == summary["checkpoint_sha256"]
    assert analyzed["actual_checkpoint_sha256"] == summary["checkpoint_sha256"]
    assert analyzed["market_compatibility"] == "verified"
    assert analyzed["can_use_for_matched_gate"] is True
    assert (output_dir / "CHECKPOINT_REPORT.md").exists()
    assert json.loads((output_dir / "checkpoint_summary.json").read_text(encoding="utf-8"))["completed_steps"] == 100


def test_large_artifact_git_attributes_are_declared() -> None:
    text = Path(".gitattributes").read_text(encoding="utf-8")
    assert "results/**/trajectory_diagnostics/*.csv linguist-generated=true" in text
    assert "results/**/trajectory_diagnostics/*.csv.gz binary" in text
    assert "results/**/checkpoints/** binary" in text
    assert "*.npz binary" in text
    assert "*.pt binary" in text
