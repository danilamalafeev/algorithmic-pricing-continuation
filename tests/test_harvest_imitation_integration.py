from __future__ import annotations

import json

from experiments.dqn_oracle_config import QVictimOracleConfig
from experiments.dqn_oracle_experiment import run_experiment
from experiments.harvest_imitation_runtime import observable_imitation_feature_dim
from neural.observations import observation_dim


def test_imitation_feature_dim_excludes_qaware_victim_internal_features():
    assert observable_imitation_feature_dim(8) == observation_dim(8)


def test_imitation_option_dqn_smoke_writes_bc_and_input_metadata(tmp_path):
    out = tmp_path / "imitation"
    result = run_experiment(
        QVictimOracleConfig(
            oracle_kind="imitation_option_dqn",
            seed=0,
            B=4,
            H=4,
            K=5,
            total_steps=20,
            eval_every=20,
            eval_steps=10,
            log_every=10,
            batch_size=4,
            train_every=1,
            target_update_every=5,
            hidden_dim=16,
            replay_capacity=100,
            imitation_bc_steps=20,
            imitation_bc_epochs=1,
            imitation_bc_batch_size=8,
            oracle_epsilon_start=0.1,
            oracle_epsilon_end=0.0,
            oracle_epsilon_decay_steps=20,
            eval_modes="continuation_adaptive",
            out_dir=str(out),
        )
    )
    summary = result["summary"]
    assert summary["oracle_kind"] == "imitation_option_dqn"
    assert summary["bc_validation_rows"] > 0
    assert 0.0 <= summary["bc_validation_accuracy"] <= 1.0
    manifest = json.loads((out / "input_manifest.json").read_text(encoding="utf-8"))
    assert manifest["feature_dim"] == observation_dim(4)
    assert "victim_q_table" in manifest["prohibited_inputs"]
    assert (out / "bc_metrics.json").exists()
