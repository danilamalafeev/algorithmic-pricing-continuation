from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.dqn_oracle_config import QVictimOracleConfig
from experiments.dqn_oracle_experiment import run_experiment


def _config(oracle_kind: str, out_dir: Path, total_steps: int) -> QVictimOracleConfig:
    return QVictimOracleConfig(
        oracle_kind=oracle_kind,
        seed=7,
        B=4,
        H=2,
        K=5,
        total_steps=total_steps,
        eval_every=4,
        eval_steps=4,
        log_every=4,
        replay_capacity=64,
        batch_size=4,
        train_every=1,
        target_update_every=3,
        oracle_epsilon_decay_steps=12,
        hidden_dim=8,
        reservoir_dim=8,
        device="cpu",
        save_final_state=True,
        imitation_bc_steps=8,
        imitation_bc_epochs=1,
        imitation_bc_batch_size=4,
        out_dir=str(out_dir),
    )


def _assert_npz_equal(left_path: Path, right_path: Path) -> None:
    with np.load(left_path) as left, np.load(right_path) as right:
        assert left.files == right.files
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key])


@pytest.mark.parametrize("oracle_kind", ["dqn", "imitation_option_dqn"])
def test_resume_matches_uninterrupted_training(tmp_path: Path, oracle_kind: str) -> None:
    full_dir = tmp_path / "full"
    split_dir = tmp_path / "split"
    full = run_experiment(_config(oracle_kind, full_dir, total_steps=12))

    first_leg = replace(
        _config(oracle_kind, split_dir, total_steps=8),
        checkpoint_every=4,
    )
    run_experiment(first_leg)
    checkpoint_path = split_dir / "training_checkpoint.pt"
    assert checkpoint_path.exists()

    resumed = run_experiment(
        replace(
            _config(oracle_kind, split_dir, total_steps=12),
            checkpoint_every=4,
            resume_from=str(checkpoint_path),
        )
    )

    pd.testing.assert_frame_equal(full["train_metrics"], resumed["train_metrics"])
    pd.testing.assert_frame_equal(full["eval_metrics"], resumed["eval_metrics"])
    _assert_npz_equal(
        full_dir / "final_victim_state.npz",
        split_dir / "final_victim_state.npz",
    )
    _assert_npz_equal(
        full_dir / "final_oracle_state.npz",
        split_dir / "final_oracle_state.npz",
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 12
    assert checkpoint["optimizer"]["state"]
    assert checkpoint["replay"]["size"] > 0
    assert checkpoint["rng_state"]["torch_generator"].numel() > 0
    assert checkpoint["environment_action_history"].shape == (4, 2, 2)
    assert not checkpoint_path.with_name(checkpoint_path.name + ".tmp").exists()


def test_resume_rejects_incompatible_training_config(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    run_experiment(
        replace(
            _config("dqn", out_dir, total_steps=4),
            checkpoint_every=4,
        )
    )

    with pytest.raises(ValueError, match="differing fields: gamma"):
        run_experiment(
            replace(
                _config("dqn", out_dir, total_steps=8),
                gamma=0.9,
                resume_from=str(out_dir / "training_checkpoint.pt"),
            )
        )
