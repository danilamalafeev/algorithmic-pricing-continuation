from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from neural.observations import ObservationConfig

from experiments.dqn_oracle_vs_qvictim import (
    DQNOracleConfig,
    clone_victim_state,
    clone_params,
    dqn_forward,
    dqn_jepa_train_step,
    dqn_regret_train_step,
    dqn_shared_jepa_train_step,
    dqn_train_step,
    dqn_victim_aware_train_step,
    evaluate,
    init_dqn_params,
    init_jepa_params,
    init_regret_params,
    init_shared_jepa_params,
    init_victim_aware_params,
    init_scripted_teacher_state,
    init_static_cooperative_victim_state,
    init_tabular_teacher_oracle_state,
    init_tabular_cfr_state,
    init_replay_buffer,
    init_victim_state,
    load_victim_state,
    jepa_encode,
    jepa_predict,
    LEARNED_HARVEST_OPTION_TO_ID,
    LEARNED_HARVEST_OPTION_NAMES,
    make_calvano_vec_env,
    oracle_counterfactual_profit,
    oracle_dqn_forward_for_kind,
    oracle_epsilon,
    parse_args as parse_oracle_args,
    qaware_option_feature_dim,
    qaware_option_features,
    replay_add,
    replay_sample,
    regret_forward,
    run_dqn_oracle_vs_qvictim,
    scripted_oracle_select_actions,
    shared_jepa_forward,
    tabular_cfr_counterfactual_next_state_ids,
    tabular_cfr_select_actions,
    tabular_cfr_state_id,
    tabular_cfr_update,
    tabular_multi_cfr_cf_value,
    tabular_multi_cfr_value_update,
    tabular_lola_select_actions,
    tabular_model_lola_select_actions,
    tabular_model_lola_values,
    tabular_rollout_lola_select_actions,
    tabular_rollout_lola_values,
    tabular_rollout_lola_values_torch,
    tabular_teacher_anchor_action,
    tabular_teacher_option_actions,
    tabular_teacher_option_durations_from_config,
    tabular_teacher_select_new_options,
    tabular_teacher_should_terminate,
    tabular_teacher_state_id,
    tabular_teacher_update_on_termination,
    teacher_epsilon,
    TEACHER_OPTION_NAMES,
    TEACHER_OPTION_TO_ID,
    victim_policy_probs_from_q,
    victim_policy_from_q,
    update_victim_q,
    victim_aware_forward,
    victim_q_update,
    victim_select_actions,
    victim_state_sha256,
)


def test_victim_state_shapes():
    B, K = 4, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    state = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    assert state["Q"].shape == (B, K * K, K)
    assert state["state_id"].shape == (B,)


def test_victim_q_initialization_eq8():
    B, K = 3, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    state = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    for av in range(K):
        expected = np.mean(profit_matrix[:, av, 1]) / (1.0 - 0.95)
        np.testing.assert_allclose(state["Q"][:, :, av], expected)


def test_victim_action_selection():
    B, K = 4, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    state = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    state["Q"][:] = 1.0
    actions = victim_select_actions(state, K, beta=0.0, rng=np.random.default_rng(2), epsilon_override=0.0)
    assert np.all(actions == 0)
    random_actions = victim_select_actions(state, K, beta=0.0, rng=np.random.default_rng(2), epsilon_override=1.0)
    assert random_actions.shape == (B,)
    assert np.all((0 <= random_actions) & (random_actions < K))


def test_clone_victim_state_copies_arrays():
    B, K = 2, 4
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    state = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    cloned = clone_victim_state(state)
    cloned["Q"][0, 0, 0] += 1.0
    cloned["state_id"][0] += 1
    cloned["t"][0] += 1
    assert not np.array_equal(cloned["Q"], state["Q"])
    assert not np.array_equal(cloned["state_id"], state["state_id"])
    assert not np.array_equal(cloned["t"], state["t"])


def test_victim_state_round_trip_and_hash_validation(tmp_path):
    B, K = 3, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    state = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    state["t"][:] = 123
    path = tmp_path / "victim.npz"
    np.savez_compressed(path, **state)

    digest = victim_state_sha256(path)
    loaded, provenance = load_victim_state(
        path,
        B=B,
        K=K,
        expected_kind="adaptive_q",
        expected_sha256=digest,
    )

    np.testing.assert_array_equal(loaded["Q"], state["Q"])
    np.testing.assert_array_equal(loaded["state_id"], state["state_id"])
    np.testing.assert_array_equal(loaded["t"], state["t"])
    assert provenance["sha256"] == digest
    assert provenance["t_min"] == 123
    loaded["Q"][0, 0, 0] += 1.0
    assert not np.array_equal(loaded["Q"], state["Q"])


def test_victim_state_rejects_wrong_shape_and_hash(tmp_path):
    path = tmp_path / "victim.npz"
    np.savez_compressed(
        path,
        kind=np.asarray("adaptive_q"),
        Q=np.zeros((2, 25, 5), dtype=np.float64),
        state_id=np.zeros(2, dtype=np.int64),
        t=np.zeros(2, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_victim_state(
            path,
            B=2,
            K=5,
            expected_kind="adaptive_q",
            expected_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="must have shape"):
        load_victim_state(path, B=3, K=5, expected_kind="adaptive_q")


def test_experiment_restores_initial_victim_and_writes_provenance(tmp_path):
    B, K = 3, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    state = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    state["t"][:] = 10_000
    source = tmp_path / "source.npz"
    np.savez_compressed(source, **state)
    out = tmp_path / "run"

    result = run_dqn_oracle_vs_qvictim(
        DQNOracleConfig(
            oracle_kind="dqn",
            seed=0,
            B=B,
            H=4,
            K=K,
            total_steps=4,
            eval_every=4,
            eval_steps=3,
            batch_size=4,
            train_every=2,
            hidden_dim=8,
            replay_capacity=20,
            initial_victim_state_mode="mature_initialization",
            initial_victim_state_path=str(source),
            out_dir=str(out),
        )
    )

    digest = victim_state_sha256(source)
    assert result["summary"]["initial_victim_state_sha256"] == digest
    assert result["summary"]["initial_victim_t_min"] == 10_000
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (out / "initial_victim_state.json").read_text(encoding="utf-8")
    )
    assert config["initial_victim_state_sha256"] == digest
    assert provenance["sha256"] == digest


def test_frozen_greedy_training_keeps_mature_q_table_and_clock_fixed(tmp_path):
    B, K = 3, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    state = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    state["Q"] += np.arange(K, dtype=np.float64)[None, None, :]
    state["t"][:] = 10_000
    source = tmp_path / "source.npz"
    np.savez_compressed(source, **state)
    out = tmp_path / "run"

    result = run_dqn_oracle_vs_qvictim(
        DQNOracleConfig(
            oracle_kind="dqn",
            seed=0,
            B=B,
            H=4,
            K=K,
            total_steps=6,
            eval_every=6,
            eval_steps=3,
            batch_size=4,
            train_every=2,
            hidden_dim=8,
            replay_capacity=20,
            initial_victim_state_mode="mature_initialization",
            initial_victim_state_path=str(source),
            victim_training_mode="frozen_greedy",
            save_final_state=True,
            out_dir=str(out),
        )
    )

    with np.load(out / "final_victim_state.npz") as final:
        np.testing.assert_array_equal(final["Q"], state["Q"])
        np.testing.assert_array_equal(final["t"], state["t"])
        assert final["state_id"].shape == state["state_id"].shape
    assert result["summary"]["victim_training_mode"] == "frozen_greedy"
    assert result["train_metrics"]["victim_epsilon"].eq(0.0).all()


def test_frozen_greedy_training_requires_mature_initialization():
    with pytest.raises(ValueError, match="requires mature_initialization"):
        run_dqn_oracle_vs_qvictim(
            DQNOracleConfig(
                oracle_kind="dqn",
                B=2,
                H=4,
                K=5,
                total_steps=1,
                victim_training_mode="frozen_greedy",
            )
        )


def test_default_eval_mode_keeps_backward_compatible_columns():
    config = DQNOracleConfig(
        oracle_kind="tabular_cfr",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=6,
        eval_every=3,
        eval_steps=4,
    )
    result = run_dqn_oracle_vs_qvictim(config)
    eval_df = result["eval_metrics"]
    summary = result["summary"]
    assert "eval_avg_profit_oracle" in eval_df.columns
    assert "eval_fresh_adaptive_avg_profit_oracle" in eval_df.columns
    assert "final_eval_avg_profit_oracle" in summary
    assert "final_eval_fresh_adaptive_avg_profit_oracle" in summary


def test_multiple_eval_modes_produce_prefixed_columns():
    config = DQNOracleConfig(
        oracle_kind="tabular_cfr",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=6,
        eval_every=3,
        eval_steps=4,
        eval_modes="fresh_adaptive,continuation_adaptive,continuation_frozen_greedy",
    )
    eval_df = run_dqn_oracle_vs_qvictim(config)["eval_metrics"]
    assert "eval_fresh_adaptive_avg_profit_oracle" in eval_df.columns
    assert "eval_continuation_adaptive_avg_profit_oracle" in eval_df.columns
    assert "eval_continuation_frozen_greedy_avg_profit_oracle" in eval_df.columns


def test_learned_harvest_oracle_smoke_with_eval_modes():
    config = DQNOracleConfig(
        oracle_kind="learned_harvest_oracle",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=20,
        eval_every=10,
        eval_steps=5,
        eval_modes="fresh_adaptive,continuation_adaptive,continuation_frozen_greedy",
    )
    result = run_dqn_oracle_vs_qvictim(config)
    eval_df = result["eval_metrics"]
    assert len(eval_df) > 0
    assert "eval_fresh_adaptive_avg_profit_oracle" in eval_df.columns
    assert "eval_continuation_adaptive_avg_profit_oracle" in eval_df.columns
    assert "eval_continuation_frozen_greedy_avg_profit_oracle" in eval_df.columns
    assert "eval_fresh_adaptive_teacher_option_freq_HARVEST_UNDERCUT_1" in eval_df.columns
    assert "eval_fresh_adaptive_teacher_option_freq_HARVEST_UNDERCUT_2" in eval_df.columns
    assert "eval_fresh_adaptive_teacher_option_freq_PUNISH_NASH" in eval_df.columns
    assert "eval_fresh_adaptive_teacher_option_freq_RESET_HIGH" in eval_df.columns


def test_constrained_qaware_option_dqn_smoke_with_eval_modes():
    config = DQNOracleConfig(
        oracle_kind="constrained_qaware_option_dqn",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=30,
        eval_every=15,
        eval_steps=5,
        eval_modes=(
            "fresh_adaptive,continuation_adaptive,"
            "continuation_frozen_greedy,continuation_frozen_epsilon"
        ),
        batch_size=8,
        train_every=2,
        hidden_dim=8,
        replay_capacity=200,
    )
    result = run_dqn_oracle_vs_qvictim(config)
    eval_df = result["eval_metrics"]
    train_df = result["train_metrics"]
    summary = result["summary"]
    assert len(eval_df) > 0
    assert "eval_fresh_adaptive_avg_profit_oracle" in eval_df.columns
    assert "eval_continuation_adaptive_avg_profit_oracle" in eval_df.columns
    assert "eval_continuation_frozen_greedy_avg_profit_oracle" in eval_df.columns
    assert "eval_continuation_frozen_epsilon_avg_profit_oracle" in eval_df.columns
    assert "eval_fresh_adaptive_teacher_option_freq_HOLD_HIGH" in eval_df.columns
    assert "eval_fresh_adaptive_teacher_option_freq_MATCH_HIGH" in eval_df.columns
    assert "eval_fresh_adaptive_teacher_option_freq_HARVEST_UNDERCUT_1" in eval_df.columns
    assert "eval_fresh_adaptive_teacher_option_freq_HARVEST_UNDERCUT_2" in eval_df.columns
    for forbidden in ("PUNISH_NASH", "PUNISH_LOW", "RESET_HIGH", "REPAIR_HIGH"):
        assert forbidden not in summary["teacher_option_names"]
        assert f"teacher_option_freq_{forbidden}" not in train_df.columns
        assert f"eval_fresh_adaptive_teacher_option_freq_{forbidden}" not in eval_df.columns
        assert f"eval_continuation_adaptive_teacher_option_freq_{forbidden}" not in eval_df.columns
    assert summary["constrained_qaware_anchor"] == "q_vs_q"
    assert summary["constrained_qaware_allow_reset"] is False
    assert summary["constrained_qaware_harvest_ticks"] == "1,2"
    assert summary["constrained_qaware_allow_repair"] is False


def test_constrained_qaware_harvest_ticks_1_exposes_only_tick1():
    config = DQNOracleConfig(
        oracle_kind="constrained_qaware_option_dqn",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=10,
        eval_every=10,
        eval_steps=3,
        eval_modes="fresh_adaptive",
        batch_size=8,
        train_every=2,
        hidden_dim=8,
        replay_capacity=100,
        constrained_qaware_harvest_ticks="1",
    )
    result = run_dqn_oracle_vs_qvictim(config)
    eval_df = result["eval_metrics"]
    assert result["summary"]["teacher_option_names"] == ["HOLD_HIGH", "MATCH_HIGH", "HARVEST_UNDERCUT_1"]
    assert "eval_fresh_adaptive_teacher_option_freq_HARVEST_UNDERCUT_1" in eval_df.columns
    assert "eval_fresh_adaptive_teacher_option_freq_HARVEST_UNDERCUT_2" not in eval_df.columns


def test_constrained_qaware_harvest_ticks_2_exposes_only_tick2():
    config = DQNOracleConfig(
        oracle_kind="constrained_qaware_option_dqn",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=10,
        eval_every=10,
        eval_steps=3,
        eval_modes="fresh_adaptive",
        batch_size=8,
        train_every=2,
        hidden_dim=8,
        replay_capacity=100,
        constrained_qaware_harvest_ticks="2",
    )
    result = run_dqn_oracle_vs_qvictim(config)
    eval_df = result["eval_metrics"]
    assert result["summary"]["teacher_option_names"] == ["HOLD_HIGH", "MATCH_HIGH", "HARVEST_UNDERCUT_2"]
    assert "eval_fresh_adaptive_teacher_option_freq_HARVEST_UNDERCUT_1" not in eval_df.columns
    assert "eval_fresh_adaptive_teacher_option_freq_HARVEST_UNDERCUT_2" in eval_df.columns


def test_constrained_qaware_repair_option_logs_eval_metrics():
    config = DQNOracleConfig(
        oracle_kind="constrained_qaware_option_dqn",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=20,
        eval_every=10,
        eval_steps=4,
        eval_modes="fresh_adaptive,continuation_adaptive",
        batch_size=8,
        train_every=2,
        hidden_dim=8,
        replay_capacity=100,
        constrained_qaware_harvest_ticks="1",
        constrained_qaware_allow_repair=True,
    )
    result = run_dqn_oracle_vs_qvictim(config)
    eval_df = result["eval_metrics"]
    assert result["summary"]["teacher_option_names"] == [
        "HOLD_HIGH",
        "MATCH_HIGH",
        "HARVEST_UNDERCUT_1",
        "REPAIR_HIGH",
    ]
    assert "eval_fresh_adaptive_teacher_option_freq_REPAIR_HIGH" in eval_df.columns
    assert "eval_continuation_adaptive_teacher_option_freq_REPAIR_HIGH" in eval_df.columns
    assert "eval_fresh_adaptive_teacher_option_freq_RESET_HIGH" not in eval_df.columns


def _assert_trajectory_diagnostics_written(out_dir):
    diag_dir = out_dir / "trajectory_diagnostics"
    summary_path = diag_dir / "protocol_summary.json"
    csv_path = diag_dir / "final_eval_fresh_adaptive.csv"
    assert summary_path.exists()
    assert csv_path.exists()
    summary = json.loads(summary_path.read_text())
    assert "fresh_adaptive" in summary
    df = pd.read_csv(csv_path)
    required = {
        "eval_mode",
        "step",
        "row",
        "oracle_action",
        "victim_action",
        "oracle_price",
        "victim_price",
        "oracle_profit",
        "victim_profit",
        "oracle_option_name",
        "compliance_flag",
    }
    assert required.issubset(set(df.columns))
    assert len(df) > 0


def test_scripted_harvest_trajectory_diagnostics_smoke(tmp_path):
    out_dir = tmp_path / "scripted_diag"
    config = DQNOracleConfig(
        oracle_kind="scripted_harvest_undercut",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=10,
        eval_every=10,
        eval_steps=4,
        eval_modes="fresh_adaptive,continuation_adaptive",
        save_trajectory_diagnostics=True,
        trajectory_diagnostic_steps=2,
        out_dir=str(out_dir),
    )
    run_dqn_oracle_vs_qvictim(config)
    _assert_trajectory_diagnostics_written(out_dir)
    assert (out_dir / "trajectory_diagnostics" / "final_eval_continuation_adaptive.csv").exists()


def test_constrained_qaware_trajectory_diagnostics_smoke(tmp_path):
    out_dir = tmp_path / "constrained_diag"
    config = DQNOracleConfig(
        oracle_kind="constrained_qaware_option_dqn",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=10,
        eval_every=10,
        eval_steps=4,
        batch_size=8,
        train_every=2,
        hidden_dim=8,
        replay_capacity=100,
        eval_modes="fresh_adaptive,continuation_adaptive",
        save_trajectory_diagnostics=True,
        trajectory_diagnostic_steps=2,
        out_dir=str(out_dir),
    )
    run_dqn_oracle_vs_qvictim(config)
    _assert_trajectory_diagnostics_written(out_dir)


def test_trajectory_diagnostics_not_written_without_flag(tmp_path):
    out_dir = tmp_path / "no_diag"
    config = DQNOracleConfig(
        oracle_kind="scripted_harvest_undercut",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=10,
        eval_every=10,
        eval_steps=4,
        out_dir=str(out_dir),
    )
    run_dqn_oracle_vs_qvictim(config)
    assert not (out_dir / "trajectory_diagnostics").exists()


def test_learned_harvest_save_final_state_contains_q_meta(tmp_path):
    out_dir = tmp_path / "learned_harvest_final_state"
    config = DQNOracleConfig(
        oracle_kind="learned_harvest_oracle",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=20,
        eval_every=10,
        eval_steps=5,
        eval_modes="fresh_adaptive,continuation_adaptive",
        save_final_state=True,
        out_dir=str(out_dir),
    )
    run_dqn_oracle_vs_qvictim(config)
    assert (out_dir / "final_victim_state.npz").exists()
    oracle_state_path = out_dir / "final_oracle_state.npz"
    assert oracle_state_path.exists()
    oracle_state = np.load(oracle_state_path)
    assert "Q_meta" in oracle_state.files
    assert oracle_state["Q_meta"].shape[1] == len(LEARNED_HARVEST_OPTION_NAMES)


def test_qaware_option_dqn_smoke_with_eval_modes():
    config = DQNOracleConfig(
        oracle_kind="qaware_option_dqn",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=30,
        eval_every=15,
        eval_steps=5,
        eval_modes="fresh_adaptive,continuation_adaptive,continuation_frozen_greedy",
        batch_size=8,
        train_every=2,
        hidden_dim=8,
        replay_capacity=200,
    )
    result = run_dqn_oracle_vs_qvictim(config)
    eval_df = result["eval_metrics"]
    assert len(eval_df) > 0
    assert "eval_fresh_adaptive_avg_profit_oracle" in eval_df.columns
    assert "eval_continuation_adaptive_avg_profit_oracle" in eval_df.columns
    assert "eval_continuation_frozen_greedy_avg_profit_oracle" in eval_df.columns
    assert "eval_fresh_adaptive_teacher_option_freq_HARVEST_UNDERCUT_1" in eval_df.columns
    assert "eval_fresh_adaptive_teacher_option_freq_HARVEST_UNDERCUT_2" in eval_df.columns
    assert "eval_fresh_adaptive_teacher_option_freq_PUNISH_NASH" in eval_df.columns
    assert "eval_fresh_adaptive_teacher_option_freq_RESET_HIGH" in eval_df.columns


def test_qaware_option_dqn_save_final_state(tmp_path):
    out_dir = tmp_path / "qaware_save"
    config = DQNOracleConfig(
        oracle_kind="qaware_option_dqn",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=10,
        eval_every=10,
        eval_steps=3,
        batch_size=8,
        train_every=2,
        hidden_dim=8,
        replay_capacity=100,
        save_final_state=True,
        out_dir=str(out_dir),
    )
    run_dqn_oracle_vs_qvictim(config)
    state_path = out_dir / "final_oracle_state.npz"
    assert state_path.exists()
    state = np.load(state_path)
    assert any(name.startswith("W") or name.startswith("b") for name in state.files)


def test_irrelevant_learned_harvest_args_do_not_break_tabular_cfr():
    config = DQNOracleConfig(
        oracle_kind="tabular_cfr",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=6,
        eval_every=3,
        eval_steps=4,
        learned_harvest_aggression_actions="bad_value",
    )
    result = run_dqn_oracle_vs_qvictim(config)
    assert not result["eval_metrics"].empty


def test_learned_harvest_rejects_bad_aggression_actions():
    config = DQNOracleConfig(
        oracle_kind="learned_harvest_oracle",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=6,
        eval_every=3,
        eval_steps=4,
        learned_harvest_aggression_actions="bad_value",
    )
    with pytest.raises(ValueError):
        run_dqn_oracle_vs_qvictim(config)


def test_continuation_eval_modes_use_trained_victim_t():
    beta = 0.01
    config = DQNOracleConfig(
        oracle_kind="tabular_cfr",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=100,
        eval_every=100,
        eval_steps=5,
        victim_beta=beta,
        eval_modes="fresh_adaptive,continuation_adaptive,continuation_frozen_epsilon",
    )
    final = run_dqn_oracle_vs_qvictim(config)["eval_metrics"].iloc[-1]
    assert final["eval_fresh_adaptive_victim_avg_epsilon"] > 0.95
    expected_trained_eps = float(np.exp(-beta * config.total_steps))
    assert final["eval_continuation_frozen_epsilon_victim_avg_epsilon"] <= expected_trained_eps + 1.0e-12
    assert final["eval_continuation_adaptive_victim_avg_epsilon"] < final["eval_fresh_adaptive_victim_avg_epsilon"]


def test_frozen_greedy_eval_does_not_mutate_template_victim_q():
    B, K = 3, 5
    _, _, benchmarks, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    victim["Q"] += np.arange(K, dtype=np.float64)
    victim["t"][:] = 100
    victim_template = clone_victim_state(victim)
    before_q = victim_template["Q"].copy()
    params = init_tabular_cfr_state(B, K, "joint_last_action", torch.device("cpu"))
    config = DQNOracleConfig(
        oracle_kind="tabular_cfr",
        seed=0,
        B=B,
        H=4,
        K=K,
        eval_steps=5,
    )
    evaluate(
        config,
        params,
        {},
        benchmarks,
        victim_template=victim_template,
        eval_mode="continuation_frozen_greedy",
        freeze_victim_q=True,
        victim_greedy=True,
        use_fresh_victim=False,
    )
    np.testing.assert_allclose(victim_template["Q"], before_q)


def test_frozen_greedy_victim_selection_uses_greedy_policy():
    B, K = 4, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    victim["state_id"][:] = 0
    victim["Q"][:] = 0.0
    victim["Q"][:, 0, 3] = 10.0
    victim["t"][:] = 0
    actions = victim_select_actions(victim, K, beta=0.0, rng=np.random.default_rng(4), greedy=True)
    assert np.all(actions == 3)


def test_static_cooperative_victim_is_non_adaptive():
    B, K = 4, 5
    _, _, benchmarks, _ = make_calvano_vec_env(B, H=4, K=K, seed=0)
    state = init_static_cooperative_victim_state(B, K, benchmarks, seed=1)
    before_q = state["Q"].copy()
    actions0 = victim_select_actions(state, K, beta=0.0, rng=np.random.default_rng(2), epsilon_override=1.0)
    update_victim_q(
        state,
        oracle_actions=np.arange(B) % K,
        victim_actions=np.arange(B) % K,
        rewards_victim=np.linspace(0.0, 1.0, B),
        alpha=1.0,
        delta=0.0,
        K=K,
    )
    actions1 = victim_select_actions(state, K, beta=0.0, rng=np.random.default_rng(3), epsilon_override=1.0)
    assert np.all(actions0 == int(benchmarks.monopoly_actions[1]))
    assert np.all(actions1 == actions0)
    np.testing.assert_allclose(state["Q"], before_q)


def test_cli_default_victim_kind_is_adaptive_q(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dqn_oracle_vs_qvictim"])
    args = parse_oracle_args()
    assert args.victim_kind == "adaptive_q"
    assert args.victim_training_mode == "adaptive"


def test_cli_accepts_rollout_lola_backend_torch(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dqn_oracle_vs_qvictim", "--rollout-lola-backend", "torch"])
    args = parse_oracle_args()
    assert args.rollout_lola_backend == "torch"


def test_cli_accepts_learned_harvest_oracle(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dqn_oracle_vs_qvictim", "--oracle-kind", "learned_harvest_oracle"])
    args = parse_oracle_args()
    assert args.oracle_kind == "learned_harvest_oracle"


def test_cli_accepts_qaware_option_dqn(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dqn_oracle_vs_qvictim", "--oracle-kind", "qaware_option_dqn"])
    args = parse_oracle_args()
    assert args.oracle_kind == "qaware_option_dqn"


def test_cli_accepts_constrained_qaware_option_dqn(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dqn_oracle_vs_qvictim",
            "--oracle-kind",
            "constrained_qaware_option_dqn",
            "--constrained-qaware-anchor",
            "monopoly",
            "--constrained-qaware-harvest-ticks",
            "1",
            "--constrained-qaware-allow-reset",
            "--constrained-qaware-allow-repair",
            "--save-trajectory-diagnostics",
            "--trajectory-diagnostic-steps",
            "25",
        ],
    )
    args = parse_oracle_args()
    assert args.oracle_kind == "constrained_qaware_option_dqn"
    assert args.constrained_qaware_anchor == "monopoly"
    assert args.constrained_qaware_harvest_ticks == "1"
    assert args.constrained_qaware_allow_reset is True
    assert args.constrained_qaware_allow_repair is True
    assert args.save_trajectory_diagnostics is True
    assert args.trajectory_diagnostic_steps == 25


def test_learned_harvest_action_mapping_has_two_undercuts():
    B, K = 2, 5
    _, _, benchmarks, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    victim["state_id"][:] = 0 * K + 4
    teacher_state = init_tabular_teacher_oracle_state(
        B,
        K,
        benchmarks,
        option_names=LEARNED_HARVEST_OPTION_NAMES,
    )
    teacher_state["active_option"][:] = np.array(
        [
            LEARNED_HARVEST_OPTION_TO_ID["HARVEST_UNDERCUT_1"],
            LEARNED_HARVEST_OPTION_TO_ID["HARVEST_UNDERCUT_2"],
        ],
        dtype=np.int64,
    )
    actions = tabular_teacher_option_actions(
        teacher_state,
        victim,
        K,
        benchmarks,
        anchor_action=4,
        harvest_aggression_actions=(1, 2),
    )
    np.testing.assert_array_equal(actions, np.array([3, 2], dtype=np.int64))


def test_qaware_option_features_shape_and_finite():
    B, H, K = 3, 4, 5
    env, price_grid, benchmarks, profit_matrix = make_calvano_vec_env(B, H=H, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    features = qaware_option_features(
        env,
        ObservationConfig(price_min=float(price_grid[0]), price_max=float(price_grid[-1]), device="cpu"),
        t=0,
        victim=victim,
        K=K,
        benchmarks=benchmarks,
        anchor_action=4,
        device="cpu",
    )
    assert features.shape == (B, qaware_option_feature_dim(H))
    assert torch.isfinite(features).all()


def test_qaware_option_mapping_matches_learned_harvest_undercuts():
    B, K = 2, 5
    _, _, benchmarks, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    victim["state_id"][:] = 0 * K + 4
    qaware_state = init_tabular_teacher_oracle_state(
        B,
        K,
        benchmarks,
        option_names=LEARNED_HARVEST_OPTION_NAMES,
    )
    qaware_state["active_option"][:] = np.array(
        [
            LEARNED_HARVEST_OPTION_TO_ID["HARVEST_UNDERCUT_1"],
            LEARNED_HARVEST_OPTION_TO_ID["HARVEST_UNDERCUT_2"],
        ],
        dtype=np.int64,
    )
    qaware_actions = tabular_teacher_option_actions(
        qaware_state,
        victim,
        K,
        benchmarks,
        anchor_action=4,
        harvest_aggression_actions=(1, 2),
    )
    np.testing.assert_array_equal(qaware_actions, np.array([3, 2], dtype=np.int64))


def test_victim_policy_from_q_shape():
    B, K = 4, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    state = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    state["Q"][:] = 0.0
    state["Q"][:, :, 3] = 1.0
    actions = victim_policy_from_q(state, K, greedy=True)
    assert actions.shape == (B,)
    assert np.all(actions == 3)


def test_victim_policy_probs_from_q_greedy():
    q = np.array([[0.0, 2.0, 1.0], [3.0, 1.0, 3.0]], dtype=np.float64)
    probs = victim_policy_probs_from_q(q, K=3, mode="greedy")
    expected = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(probs, expected)


def test_victim_policy_probs_from_q_epsilon_greedy():
    q = np.array([[[0.0, 2.0, 1.0], [3.0, 1.0, 0.0]], [[1.0, 0.0, 2.0], [0.0, 4.0, 3.0]]], dtype=np.float64)
    probs = victim_policy_probs_from_q(q, K=3, mode="epsilon_greedy", epsilon=np.array([0.3, 0.6]))
    assert probs.shape == q.shape
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0)
    np.testing.assert_allclose(probs[0, 0], np.array([0.1, 0.8, 0.1]))
    np.testing.assert_allclose(probs[1, 0], np.array([0.2, 0.2, 0.6]))


def test_victim_policy_probs_from_q_softmax():
    q = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    probs = victim_policy_probs_from_q(q, K=3, mode="softmax", tau=1.0)
    assert probs.shape == q.shape
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0)
    assert probs[0, 2] > probs[0, 1] > probs[0, 0]
    np.testing.assert_allclose(probs[1], np.full(3, 1.0 / 3.0))


def test_victim_q_update():
    B, K = 1, 3
    Q = np.zeros((B, K * K, K), dtype=np.float64)
    Q[0, 2, :] = np.array([1.0, 2.0, 3.0])
    state = {"Q": Q, "state_id": np.array([0], dtype=np.int64), "t": np.zeros(B, dtype=np.int64)}
    victim_q_update(
        state,
        state_id=np.array([0]),
        victim_actions=np.array([1]),
        rewards_victim=np.array([0.5]),
        next_state_id=np.array([2]),
        alpha=0.2,
        delta=0.9,
    )
    expected = 0.8 * 0.0 + 0.2 * (0.5 + 0.9 * 3.0)
    np.testing.assert_allclose(state["Q"][0, 0, 1], expected)
    assert state["state_id"][0] == 2


def test_tabular_teacher_meta_q_updates_only_on_termination():
    B, K = 1, 5
    _, _, benchmarks, _ = make_calvano_vec_env(B, H=4, K=K, seed=0)
    state = init_tabular_teacher_oracle_state(B, K, benchmarks)
    option_id = TEACHER_OPTION_TO_ID["HOLD_HIGH"]
    state["state_id"][0] = 2
    state["active_option"][0] = option_id
    state["option_start_state"][0] = 2
    state["option_return"][0] = 3.0
    state["option_discount"][0] = 0.25
    state["Q_meta"][2, option_id] = 1.0
    state["Q_meta"][3, :] = 4.0

    tabular_teacher_update_on_termination(
        state,
        terminated=np.array([False]),
        next_state_id=np.array([3]),
        alpha=0.5,
    )
    assert state["Q_meta"][2, option_id] == 1.0
    assert state["active_option"][0] == option_id

    tabular_teacher_update_on_termination(
        state,
        terminated=np.array([True]),
        next_state_id=np.array([3]),
        alpha=0.5,
    )
    expected_target = 3.0 + 0.25 * 4.0
    expected = 0.5 * 1.0 + 0.5 * expected_target
    np.testing.assert_allclose(state["Q_meta"][2, option_id], expected)
    assert state["active_option"][0] == -1
    assert state["last_option"][0] == option_id


def test_tabular_teacher_selects_options_not_prices():
    B, K = 6, 7
    rng = np.random.default_rng(3)
    _, _, benchmarks, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    state = init_tabular_teacher_oracle_state(B, K, benchmarks)
    state["state_id"] = tabular_teacher_state_id(
        victim["state_id"] // K,
        victim,
        K,
        benchmarks,
    )
    transition_counts = tabular_teacher_select_new_options(state, epsilon=1.0, rng=rng)
    assert state["active_option"].shape == (B,)
    assert np.all((0 <= state["active_option"]) & (state["active_option"] < len(TEACHER_OPTION_NAMES)))
    assert transition_counts.shape == (len(TEACHER_OPTION_NAMES), len(TEACHER_OPTION_NAMES))

    actions = tabular_teacher_option_actions(state, victim, K, benchmarks)
    assert actions.shape == (B,)
    assert np.all((0 <= actions) & (actions < K))


def test_tabular_teacher_custom_durations_are_respected():
    B, K = 2, 7
    _, _, benchmarks, _ = make_calvano_vec_env(B, H=4, K=K, seed=0)
    config = DQNOracleConfig(
        teacher_hold_duration=9,
        teacher_harvest_duration=8,
        teacher_match_duration=7,
        teacher_punish_nash_duration=6,
        teacher_punish_low_duration=5,
        teacher_reset_duration=4,
    )
    durations = tabular_teacher_option_durations_from_config(config)
    assert durations[TEACHER_OPTION_TO_ID["HOLD_HIGH"]] == 9
    assert durations[TEACHER_OPTION_TO_ID["RESET_HIGH"]] == 4

    state = init_tabular_teacher_oracle_state(B, K, benchmarks)
    state["active_option"][:] = TEACHER_OPTION_TO_ID["HOLD_HIGH"]
    state["option_elapsed"][:] = np.array([8, 9], dtype=np.int64)
    victim_actions = np.full(B, int(benchmarks.monopoly_actions[1]), dtype=np.int64)
    terminated = tabular_teacher_should_terminate(state, victim_actions, benchmarks, durations)
    assert terminated.tolist() == [False, True]


def test_tabular_teacher_epsilon_schedule_is_independent_from_oracle_epsilon():
    config = DQNOracleConfig(
        oracle_epsilon_start=1.0,
        oracle_epsilon_end=0.5,
        oracle_epsilon_decay_steps=10,
        teacher_epsilon_start=0.30,
        teacher_epsilon_end=0.02,
        teacher_epsilon_decay_steps=30,
    )
    assert teacher_epsilon(config, 0) == 0.30
    assert oracle_epsilon(config, 0) == 1.0
    assert teacher_epsilon(config, 15) != oracle_epsilon(config, 15)
    np.testing.assert_allclose(teacher_epsilon(config, 30), 0.02)


def test_tabular_teacher_q_vs_q_anchor_selects_nearest_target_price():
    B, K = 1, 15
    _, price_grid, benchmarks, _ = make_calvano_vec_env(B, H=4, K=K, seed=0)
    action = tabular_teacher_anchor_action(price_grid, benchmarks, "q_vs_q")
    expected = int(np.argmin(np.abs(price_grid - 1.80)))
    assert action == expected
    assert abs(float(price_grid[action]) - 1.80) <= float(np.min(np.abs(price_grid - 1.80))) + 1e-12


def test_tabular_teacher_reset_high_ramps_monotone_nondecreasing():
    B, K = 1, 9
    _, _, benchmarks, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    state = init_tabular_teacher_oracle_state(B, K, benchmarks, anchor_action=7)
    state["active_option"][0] = TEACHER_OPTION_TO_ID["RESET_HIGH"]
    state["option_start_action"][0] = 2
    durations = np.full(len(TEACHER_OPTION_NAMES), 4, dtype=np.int64)

    actions = []
    for elapsed in range(4):
        state["option_elapsed"][0] = elapsed
        action = tabular_teacher_option_actions(
            state,
            victim,
            K,
            benchmarks,
            anchor_action=7,
            durations=durations,
        )[0]
        actions.append(int(action))

    assert actions == sorted(actions)
    assert actions[-1] == 7
    assert actions[0] > 2


def test_scripted_hold_high_always_plays_anchor_action():
    B, K = 4, 15
    _, price_grid, benchmarks, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    anchor = tabular_teacher_anchor_action(price_grid, benchmarks, "q_vs_q")
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    state = init_scripted_teacher_state(B, K, benchmarks, price_grid, anchor)
    actions = scripted_oracle_select_actions(
        state,
        "scripted_hold_high",
        victim,
        K,
        benchmarks,
        anchor,
        tabular_teacher_option_durations_from_config(DQNOracleConfig()),
    )
    assert np.all(actions == anchor)
    assert np.all(state["active_option"] == TEACHER_OPTION_TO_ID["HOLD_HIGH"])


def test_scripted_punish_nash_always_plays_nash():
    B, K = 4, 15
    _, price_grid, benchmarks, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    anchor = tabular_teacher_anchor_action(price_grid, benchmarks, "q_vs_q")
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    state = init_scripted_teacher_state(B, K, benchmarks, price_grid, anchor)
    actions = scripted_oracle_select_actions(
        state,
        "scripted_punish_nash",
        victim,
        K,
        benchmarks,
        anchor,
        tabular_teacher_option_durations_from_config(DQNOracleConfig()),
    )
    assert np.all(actions == int(benchmarks.nash_actions[0]))
    assert np.all(state["active_option"] == TEACHER_OPTION_TO_ID["PUNISH_NASH"])


def test_scripted_harvest_undercut_clips_correctly():
    B, K = 3, 15
    _, price_grid, benchmarks, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    anchor = tabular_teacher_anchor_action(price_grid, benchmarks, "q_vs_q")
    nash = int(benchmarks.nash_actions[0])
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    victim["state_id"] = np.array([0 * K + nash, 0 * K + anchor, 0 * K + (K - 1)], dtype=np.int64)
    state = init_scripted_teacher_state(B, K, benchmarks, price_grid, anchor)
    actions = scripted_oracle_select_actions(
        state,
        "scripted_harvest_undercut",
        victim,
        K,
        benchmarks,
        anchor,
        tabular_teacher_option_durations_from_config(DQNOracleConfig()),
        aggression_ticks=2,
    )
    np.testing.assert_array_equal(actions, np.array([nash, max(nash, anchor - 2), anchor], dtype=np.int64))
    assert np.all(state["active_option"] == TEACHER_OPTION_TO_ID["HARVEST_UNDERCUT"])


def test_scripted_carrot_stick_switches_to_punish_on_defection():
    B, K = 2, 15
    _, price_grid, benchmarks, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    anchor = tabular_teacher_anchor_action(price_grid, benchmarks, "q_vs_q")
    nash = int(benchmarks.nash_actions[0])
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    victim["state_id"] = np.array([0 * K + nash, 0 * K + (nash - 1)], dtype=np.int64)
    state = init_scripted_teacher_state(B, K, benchmarks, price_grid, anchor)
    state["rolling_compliance"][:] = 0.1
    actions = scripted_oracle_select_actions(
        state,
        "scripted_carrot_stick",
        victim,
        K,
        benchmarks,
        anchor,
        tabular_teacher_option_durations_from_config(DQNOracleConfig()),
        compliance_low=0.35,
        compliance_high=0.65,
        punish_mode="nash",
    )
    assert np.all(state["active_option"] == TEACHER_OPTION_TO_ID["PUNISH_NASH"])
    assert np.all(actions == nash)


def test_scripted_compliance_harvest_switches_to_harvest_after_compliance_high():
    B, K = 2, 15
    _, price_grid, benchmarks, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    anchor = tabular_teacher_anchor_action(price_grid, benchmarks, "q_vs_q")
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    victim["state_id"] = np.array([0 * K + anchor, 0 * K + anchor], dtype=np.int64)
    state = init_scripted_teacher_state(B, K, benchmarks, price_grid, anchor)
    state["rolling_compliance"][:] = 0.8
    actions = scripted_oracle_select_actions(
        state,
        "scripted_compliance_harvest",
        victim,
        K,
        benchmarks,
        anchor,
        tabular_teacher_option_durations_from_config(DQNOracleConfig()),
        compliance_low=0.35,
        compliance_high=0.65,
    )
    assert np.all(state["active_option"] == TEACHER_OPTION_TO_ID["HARVEST_UNDERCUT"])
    assert np.all(actions == anchor - 1)


def test_dqn_forward_shape():
    B, Z, K = 4, 11, 5
    gen = torch.Generator().manual_seed(3)
    params = init_dqn_params(gen, Z, hidden_dim=7, K=K)
    q = dqn_forward(params, torch.randn(B, Z))
    assert q.shape == (B, K)


def test_replay_buffer_add_sample():
    B, Z, K = 4, 6, 5
    buffer = init_replay_buffer(capacity=20, obs_dim=Z, K=K)
    obs = torch.randn(B, Z)
    next_obs = torch.randn(B, Z)
    action = torch.randint(0, K, (B,))
    victim_action = torch.randint(0, K, (B,))
    reward = torch.randn(B)
    done = torch.zeros(B, dtype=torch.bool)
    cf_profit = torch.randn(B, K)
    replay_add(buffer, obs, action, reward, next_obs, done, victim_action, cf_profit)
    assert buffer["size"] == B
    batch = replay_sample(buffer, batch_size=3, generator=torch.Generator().manual_seed(4))
    assert batch["obs"].shape == (3, Z)
    assert batch["action"].shape == (3,)
    assert batch["victim_action"].shape == (3,)
    assert batch["cf_profit"].shape == (3, K)
    assert batch["victim_greedy_action"].shape == (3,)
    assert batch["victim_anchor_action"].shape == (3,)
    assert batch["victim_anchor_compliance"].shape == (3,)
    assert batch["victim_anchor_q_gap"].shape == (3,)
    assert batch["market_aux_target"].shape == (3, 3)


def test_replay_buffer_accepts_victim_aux_fields():
    B, Z, K = 4, 6, 5
    buffer = init_replay_buffer(capacity=20, obs_dim=Z, K=K)
    replay_add(
        buffer,
        torch.randn(B, Z),
        torch.randint(0, K, (B,)),
        torch.randn(B),
        torch.randn(B, Z),
        torch.zeros(B, dtype=torch.bool),
        victim_action=torch.randint(0, K, (B,)),
        cf_profit=torch.randn(B, K),
        victim_greedy_action=torch.randint(0, K, (B,)),
        victim_anchor_action=torch.full((B,), 3, dtype=torch.long),
        victim_anchor_compliance=torch.ones(B),
        victim_anchor_q_gap=torch.linspace(0.0, 1.0, B),
        market_aux_target=torch.randn(B, 3),
    )
    batch = replay_sample(buffer, batch_size=3, generator=torch.Generator().manual_seed(44))
    assert torch.all(batch["victim_anchor_action"] == 3)
    assert torch.all(batch["victim_anchor_compliance"] == 1.0)


def test_dqn_train_step():
    B, Z, K = 8, 6, 5
    gen = torch.Generator().manual_seed(5)
    params = init_dqn_params(gen, Z, hidden_dim=7, K=K)
    target = {k: v.detach().clone() for k, v in params.items()}
    before = {k: v.detach().clone() for k, v in params.items()}
    optimizer = torch.optim.Adam(list(params.values()), lr=1e-2)
    batch = {
        "obs": torch.randn(B, Z),
        "action": torch.randint(0, K, (B,)),
        "reward": torch.randn(B),
        "next_obs": torch.randn(B, Z),
        "done": torch.zeros(B, dtype=torch.bool),
    }
    loss = dqn_train_step(params, target, batch, optimizer, gamma=0.95)
    assert np.isfinite(loss)
    assert any(not torch.allclose(before[k], params[k]) for k in params)


def test_jepa_forward_shapes():
    B, Z, H, L, K = 4, 6, 7, 5, 3
    gen = torch.Generator().manual_seed(6)
    params = init_jepa_params(gen, Z, hidden_dim=H, latent_dim=L, K=K)
    obs = torch.randn(B, Z)
    action = torch.randint(0, K, (B,))
    latent = jepa_encode(params, obs)
    pred = jepa_predict(params, latent, action, K)
    assert latent.shape == (B, L)
    assert pred.shape == (B, L)


def test_regret_forward_shape():
    B, Z, K = 4, 6, 5
    gen = torch.Generator().manual_seed(8)
    params = init_regret_params(gen, Z, hidden_dim=7, K=K)
    pred = regret_forward(params, torch.randn(B, Z))
    assert pred.shape == (B, K)


def test_tabular_cfr_state_shapes():
    device = torch.device("cpu")
    state_v = init_tabular_cfr_state(B=4, K=5, state_mode="victim_last_action", device=device)
    assert state_v["regret_table"].shape == (4, 5, 5)
    assert state_v["value_table"].shape == (4, 5)
    assert state_v["state_id"].shape == (4,)
    state_j = init_tabular_cfr_state(B=4, K=5, state_mode="joint_last_action", device=device)
    assert state_j["regret_table"].shape == (4, 25, 5)
    assert state_j["value_table"].shape == (4, 25)
    assert state_j["state_id"].shape == (4,)


def test_tabular_multi_cfr_state_has_value_table():
    state = init_tabular_cfr_state(B=3, K=4, state_mode="joint_last_action", device=torch.device("cpu"))
    assert "value_table" in state
    assert state["value_table"].shape == (3, 16)
    assert torch.all(state["value_table"] == 0.0)


def test_tabular_cfr_state_id_victim_last_action():
    state_id = tabular_cfr_state_id(
        oracle_actions=np.array([0, 2, 4]),
        victim_actions=np.array([1, 3, 0]),
        K=5,
        state_mode="victim_last_action",
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(state_id, torch.tensor([1, 3, 0], dtype=torch.long))


def test_tabular_cfr_state_id_joint_last_action():
    state_id = tabular_cfr_state_id(
        oracle_actions=np.array([0, 2, 4]),
        victim_actions=np.array([1, 3, 0]),
        K=5,
        state_mode="joint_last_action",
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(state_id, torch.tensor([1, 13, 20], dtype=torch.long))


def test_tabular_cfr_select_actions_uniform_when_zero_regret():
    B, K = 5000, 5
    state = init_tabular_cfr_state(B=B, K=K, state_mode="victim_last_action", device=torch.device("cpu"))
    actions = tabular_cfr_select_actions(state, K, epsilon=0.0, generator=torch.Generator().manual_seed(10))
    counts = torch.bincount(actions, minlength=K).to(torch.float32) / B
    torch.testing.assert_close(counts, torch.full((K,), 1.0 / K), atol=0.025, rtol=0.0)


def test_tabular_cfr_update_accumulates_regret():
    state = init_tabular_cfr_state(B=2, K=3, state_mode="victim_last_action", device=torch.device("cpu"))
    prev_state_id = torch.tensor([0, 1], dtype=torch.long)
    oracle_actions = torch.tensor([1, 2], dtype=torch.long)
    cf_profit = torch.tensor([[1.0, 2.0, 0.5], [0.3, 0.4, 0.1]], dtype=torch.float32)
    tabular_cfr_update(state, prev_state_id, oracle_actions, cf_profit, regret_decay=1.0)
    expected0 = torch.tensor([-1.0, 0.0, -1.5])
    expected1 = torch.tensor([0.2, 0.3, 0.0])
    torch.testing.assert_close(state["regret_table"][0, 0], expected0)
    torch.testing.assert_close(state["regret_table"][1, 1], expected1)
    tabular_cfr_update(state, prev_state_id, oracle_actions, cf_profit, regret_decay=1.0)
    torch.testing.assert_close(state["regret_table"][0, 0], 2.0 * expected0)
    torch.testing.assert_close(state["regret_table"][1, 1], 2.0 * expected1)


def test_counterfactual_next_state_ids_joint_last_action():
    next_state = tabular_cfr_counterfactual_next_state_ids(
        torch.arange(4),
        victim_actions=np.array([1, 3]),
        K=4,
        state_mode="joint_last_action",
        device=torch.device("cpu"),
    )
    expected = torch.tensor([[1, 5, 9, 13], [3, 7, 11, 15]], dtype=torch.long)
    torch.testing.assert_close(next_state, expected)


def test_counterfactual_next_state_ids_victim_last_action():
    next_state = tabular_cfr_counterfactual_next_state_ids(
        torch.arange(4),
        victim_actions=np.array([1, 3]),
        K=4,
        state_mode="victim_last_action",
        device=torch.device("cpu"),
    )
    expected = torch.tensor([[1, 1, 1, 1], [3, 3, 3, 3]], dtype=torch.long)
    torch.testing.assert_close(next_state, expected)


def test_tabular_multi_cfr_cf_value_shape():
    state = init_tabular_cfr_state(B=2, K=3, state_mode="victim_last_action", device=torch.device("cpu"))
    state["value_table"][0] = torch.tensor([0.0, 1.0, 2.0])
    state["value_table"][1] = torch.tensor([3.0, 4.0, 5.0])
    cf_profit = torch.ones(2, 3)
    next_state_cf = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)
    cf_value = tabular_multi_cfr_cf_value(state, cf_profit, next_state_cf, gamma=0.5)
    assert cf_value.shape == (2, 3)
    expected = torch.tensor([[1.0, 1.5, 2.0], [3.5, 3.0, 2.5]])
    torch.testing.assert_close(cf_value, expected)


def test_tabular_multi_cfr_value_update():
    state = init_tabular_cfr_state(B=2, K=3, state_mode="victim_last_action", device=torch.device("cpu"))
    state["value_table"][0, 2] = 4.0
    state["value_table"][1, 1] = 2.0
    tabular_multi_cfr_value_update(
        state,
        prev_state_id=torch.tensor([0, 1]),
        rewards_oracle=torch.tensor([1.0, 2.0]),
        next_state_real=torch.tensor([2, 1]),
        value_lr=0.5,
        gamma=0.25,
    )
    torch.testing.assert_close(state["value_table"][0, 0], torch.tensor(1.0))
    torch.testing.assert_close(state["value_table"][1, 1], torch.tensor(2.25))


def test_tabular_lola_select_actions_shape():
    B, K = 4, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    actions, metrics = tabular_lola_select_actions(
        victim,
        profit_matrix,
        K,
        gamma=0.95,
        tau=0.05,
        epsilon=0.05,
        generator=torch.Generator().manual_seed(11),
        device=torch.device("cpu"),
    )
    assert actions.shape == (B,)
    assert torch.all((0 <= actions) & (actions < K))
    assert {"lola_immediate_value", "lola_future_value", "lola_total_value", "lola_entropy"}.issubset(metrics)


def test_tabular_lola_select_actions_probs_finite():
    B, K = 4, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    _, metrics = tabular_lola_select_actions(
        victim,
        profit_matrix,
        K,
        gamma=0.95,
        tau=0.05,
        epsilon=0.05,
        generator=torch.Generator().manual_seed(12),
        device=torch.device("cpu"),
    )
    assert all(np.isfinite(v) for v in metrics.values())


def test_tabular_model_lola_values_shape():
    B, K = 4, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    values, metrics = tabular_model_lola_values(
        victim,
        profit_matrix,
        K,
        alpha=0.15,
        delta=0.95,
        beta=4e-6,
        gamma_lola=0.95,
        victim_policy_mode="epsilon_greedy",
        future_policy_mode="epsilon_greedy",
        victim_softmax_tau=0.05,
    )
    assert values.shape == (B, K)
    assert {
        "model_lola_immediate_value",
        "model_lola_future_value",
        "model_lola_total_value",
        "model_lola_current_victim_entropy",
        "model_lola_future_victim_entropy",
    }.issubset(metrics)


def test_tabular_model_lola_values_finite():
    B, K = 4, 5
    _, _, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    values, metrics = tabular_model_lola_values(
        victim,
        profit_matrix,
        K,
        alpha=0.15,
        delta=0.95,
        beta=4e-6,
        gamma_lola=0.95,
        victim_policy_mode="softmax",
        future_policy_mode="softmax",
        victim_softmax_tau=0.05,
    )
    assert np.isfinite(values).all()
    assert all(np.isfinite(v) for v in metrics.values())


def test_tabular_model_lola_select_actions_shape():
    values = np.random.default_rng(0).normal(size=(4, 5))
    actions, metrics = tabular_model_lola_select_actions(
        values,
        tau=0.05,
        epsilon=0.02,
        generator=torch.Generator().manual_seed(13),
        device=torch.device("cpu"),
    )
    assert actions.shape == (4,)
    assert torch.all((0 <= actions) & (actions < 5))
    assert {"model_lola_entropy", "model_lola_value_mean", "model_lola_value_std"}.issubset(metrics)
    assert all(np.isfinite(v) for v in metrics.values())


def test_tabular_rollout_lola_values_shape():
    B, K = 3, 4
    _, price_grid, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    values, metrics = tabular_rollout_lola_values(
        victim,
        profit_matrix,
        K,
        alpha=0.15,
        delta=0.95,
        beta=4e-6,
        horizon=3,
        num_particles=2,
        victim_policy_mode="epsilon_greedy",
        oracle_rollout_policy="greedy_best_response",
        discount=0.95,
        include_immediate=True,
        rng=np.random.default_rng(14),
        price_grid=price_grid,
    )
    assert values.shape == (B, K)
    assert {
        "rollout_lola_first_step_profit",
        "rollout_lola_future_profit",
        "rollout_lola_victim_price_simulated",
        "rollout_lola_oracle_price_simulated",
    }.issubset(metrics)


def test_tabular_rollout_lola_values_finite():
    B, K = 3, 4
    _, price_grid, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    values, metrics = tabular_rollout_lola_values(
        victim,
        profit_matrix,
        K,
        alpha=0.15,
        delta=0.95,
        beta=4e-6,
        horizon=4,
        num_particles=3,
        victim_policy_mode="epsilon_greedy",
        oracle_rollout_policy="fixed_first_action",
        discount=0.95,
        include_immediate=False,
        rng=np.random.default_rng(15),
        price_grid=price_grid,
    )
    assert np.isfinite(values).all()
    assert all(np.isfinite(v) for v in metrics.values())


def test_tabular_rollout_lola_values_torch_shape_finite_cpu():
    B, K = 3, 4
    _, price_grid, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    values, metrics = tabular_rollout_lola_values_torch(
        victim,
        profit_matrix,
        K,
        alpha=0.15,
        delta=0.95,
        beta=4e-6,
        horizon=3,
        num_particles=2,
        victim_policy_mode="epsilon_greedy",
        oracle_rollout_policy="greedy_best_response",
        discount=0.95,
        include_immediate=True,
        generator=torch.Generator().manual_seed(17),
        device=torch.device("cpu"),
        price_grid=price_grid,
    )
    assert values.shape == (B, K)
    assert torch.isfinite(values).all()
    assert {
        "rollout_lola_first_step_profit",
        "rollout_lola_future_profit",
        "rollout_lola_victim_price_simulated",
        "rollout_lola_oracle_price_simulated",
    }.issubset(metrics)
    assert all(np.isfinite(v) for v in metrics.values())


def test_tabular_rollout_lola_values_torch_reports_chunked_memory_reduction():
    B, K = 3, 5
    _, price_grid, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    values, metrics = tabular_rollout_lola_values_torch(
        victim,
        profit_matrix,
        K,
        alpha=0.15,
        delta=0.95,
        beta=4e-6,
        horizon=3,
        num_particles=2,
        victim_policy_mode="epsilon_greedy",
        oracle_rollout_policy="greedy_best_response",
        discount=0.95,
        include_immediate=True,
        generator=torch.Generator().manual_seed(117),
        device=torch.device("cpu"),
        price_grid=price_grid,
        chunk_size=2,
    )
    assert values.shape == (B, K)
    assert torch.isfinite(values).all()
    assert metrics["rollout_lola_chunked_q_clone_elements"] < metrics["rollout_lola_dense_q_clone_elements"]
    assert metrics["rollout_lola_q_clone_reduction"] == pytest.approx(K / 2)
    assert all(np.isfinite(v) for v in metrics.values())


def test_tabular_rollout_lola_values_torch_matches_numpy_greedy_tiny():
    B, K = 2, 4
    _, price_grid, _, profit_matrix = make_calvano_vec_env(B, H=4, K=K, seed=0)
    victim = init_victim_state(B, K, profit_matrix, delta=0.95, seed=1)
    numpy_values, numpy_metrics = tabular_rollout_lola_values(
        victim,
        profit_matrix,
        K,
        alpha=0.15,
        delta=0.95,
        beta=4e-6,
        horizon=3,
        num_particles=3,
        victim_policy_mode="greedy",
        oracle_rollout_policy="fixed_first_action",
        discount=0.95,
        include_immediate=True,
        rng=np.random.default_rng(18),
        price_grid=price_grid,
    )
    torch_values, torch_metrics = tabular_rollout_lola_values_torch(
        victim,
        profit_matrix,
        K,
        alpha=0.15,
        delta=0.95,
        beta=4e-6,
        horizon=3,
        num_particles=3,
        victim_policy_mode="greedy",
        oracle_rollout_policy="fixed_first_action",
        discount=0.95,
        include_immediate=True,
        generator=torch.Generator().manual_seed(18),
        device=torch.device("cpu"),
        price_grid=price_grid,
    )
    np.testing.assert_allclose(torch_values.detach().cpu().numpy(), numpy_values, rtol=1e-5, atol=1e-5)
    for key, numpy_value in numpy_metrics.items():
        np.testing.assert_allclose(torch_metrics[key], numpy_value, rtol=1e-5, atol=1e-5)


def test_tabular_rollout_lola_select_actions_shape():
    values = np.random.default_rng(0).normal(size=(4, 5))
    actions, metrics = tabular_rollout_lola_select_actions(
        values,
        tau=0.05,
        epsilon=0.02,
        generator=torch.Generator().manual_seed(16),
        device=torch.device("cpu"),
        price_grid=np.linspace(1.0, 2.0, 5),
    )
    assert actions.shape == (4,)
    assert torch.all((0 <= actions) & (actions < 5))
    assert {"rollout_lola_value_mean", "rollout_lola_value_std", "rollout_lola_entropy", "rollout_lola_best_action_price"}.issubset(metrics)
    assert all(np.isfinite(v) for v in metrics.values())


def test_oracle_counterfactual_profit_shape():
    K = 4
    profit_matrix = np.zeros((K, K, 2), dtype=np.float32)
    for a_o in range(K):
        for a_v in range(K):
            profit_matrix[a_o, a_v, 0] = 10 * a_o + a_v
    cf = oracle_counterfactual_profit(profit_matrix, np.array([0, 2, 3]), torch.device("cpu"))
    assert cf.shape == (3, K)
    torch.testing.assert_close(cf[1], torch.tensor([2.0, 12.0, 22.0, 32.0]))


def test_dqn_jepa_train_step_updates_params():
    B, Z, K = 8, 6, 5
    gen = torch.Generator().manual_seed(7)
    q_params = init_dqn_params(gen, Z, hidden_dim=7, K=K)
    target_q_params = clone_params(q_params)
    jepa_params = init_jepa_params(gen, Z, hidden_dim=7, latent_dim=4, K=K)
    target_jepa_params = clone_params(jepa_params)
    before_q = {k: v.detach().clone() for k, v in q_params.items()}
    before_jepa = {k: v.detach().clone() for k, v in jepa_params.items()}
    optimizer = torch.optim.Adam(list(q_params.values()) + list(jepa_params.values()), lr=1e-2)
    batch = {
        "obs": torch.randn(B, Z),
        "action": torch.randint(0, K, (B,)),
        "reward": torch.randn(B),
        "next_obs": torch.randn(B, Z),
        "done": torch.zeros(B, dtype=torch.bool),
    }
    metrics = dqn_jepa_train_step(
        q_params,
        target_q_params,
        jepa_params,
        target_jepa_params,
        batch,
        optimizer,
        gamma=0.95,
        jepa_coef=0.1,
        K=K,
    )
    assert set(metrics) == {"dqn_loss", "jepa_loss", "total_loss", "q_mean", "q_max"}
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.allclose(before_q[k], q_params[k]) for k in q_params)
    assert any(not torch.allclose(before_jepa[k], jepa_params[k]) for k in jepa_params)


def test_dqn_shared_jepa_loss_changes_shared_encoder_beyond_dqn_loss():
    B, Z, K = 8, 6, 5
    gen = torch.Generator().manual_seed(107)
    params_zero = init_shared_jepa_params(gen, Z, hidden_dim=7, latent_dim=4, K=K)
    target_zero = clone_params(params_zero)
    params_jepa = {k: v.detach().clone().requires_grad_(True) for k, v in params_zero.items()}
    target_jepa = clone_params(params_jepa)
    batch = {
        "obs": torch.randn(B, Z),
        "action": torch.randint(0, K, (B,)),
        "reward": torch.randn(B),
        "next_obs": torch.randn(B, Z),
        "done": torch.zeros(B, dtype=torch.bool),
    }
    opt_zero = torch.optim.SGD(list(params_zero.values()), lr=1e-2)
    opt_jepa = torch.optim.SGD(list(params_jepa.values()), lr=1e-2)
    dqn_shared_jepa_train_step(params_zero, target_zero, batch, opt_zero, gamma=0.95, jepa_coef=0.0, K=K)
    dqn_shared_jepa_train_step(params_jepa, target_jepa, batch, opt_jepa, gamma=0.95, jepa_coef=10.0, K=K)
    encoder_keys = ["enc_W1", "enc_b1", "enc_W2", "enc_b2"]
    assert any(not torch.allclose(params_zero[k], params_jepa[k]) for k in encoder_keys)


def test_dqn_shared_jepa_q_values_depend_on_shared_encoder():
    B, Z, K = 4, 6, 5
    gen = torch.Generator().manual_seed(108)
    params = init_shared_jepa_params(gen, Z, hidden_dim=7, latent_dim=4, K=K)
    obs = torch.randn(B, Z)
    q_before = shared_jepa_forward(params, obs)
    with torch.no_grad():
        params["enc_b2"].add_(1.0)
    q_after = shared_jepa_forward(params, obs)
    assert not torch.allclose(q_before, q_after)


def test_dqn_shared_jepa_zero_coef_is_shared_encoder_dqn_loss_only():
    B, Z, K = 8, 6, 5
    gen = torch.Generator().manual_seed(109)
    params = init_shared_jepa_params(gen, Z, hidden_dim=7, latent_dim=4, K=K)
    target = clone_params(params)
    before_pred = {k: params[k].detach().clone() for k in ["pred_W1", "pred_b1", "pred_W2", "pred_b2"]}
    optimizer = torch.optim.SGD(list(params.values()), lr=1e-2)
    batch = {
        "obs": torch.randn(B, Z),
        "action": torch.randint(0, K, (B,)),
        "reward": torch.randn(B),
        "next_obs": torch.randn(B, Z),
        "done": torch.zeros(B, dtype=torch.bool),
    }
    metrics = dqn_shared_jepa_train_step(params, target, batch, optimizer, gamma=0.95, jepa_coef=0.0, K=K)
    assert metrics["total_loss"] == pytest.approx(metrics["dqn_loss"])
    assert metrics["jepa_loss"] >= 0.0
    assert all(torch.allclose(before_pred[k], params[k]) for k in before_pred)


def test_dqn_shared_jepa_forward_path_is_not_decoupled_dqn():
    B, Z, K = 4, 6, 5
    gen = torch.Generator().manual_seed(110)
    params = init_shared_jepa_params(gen, Z, hidden_dim=7, latent_dim=4, K=K)
    obs = torch.randn(B, Z)
    torch.testing.assert_close(
        oracle_dqn_forward_for_kind("dqn_shared_jepa", params, obs),
        shared_jepa_forward(params, obs),
    )
    assert {"enc_W1", "enc_W2", "q_W", "q_b"}.issubset(params)
    assert "W1" not in params and "W2" not in params


def _victim_aware_batch(B: int, Z: int, K: int) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randn(B, Z),
        "action": torch.randint(0, K, (B,)),
        "victim_action": torch.randint(0, K, (B,)),
        "victim_greedy_action": torch.randint(0, K, (B,)),
        "victim_anchor_action": torch.randint(0, K, (B,)),
        "victim_anchor_compliance": torch.rand(B).round(),
        "victim_anchor_q_gap": torch.randn(B).clamp(-2.0, 2.0),
        "market_aux_target": torch.randn(B, 3),
        "reward": torch.randn(B),
        "next_obs": torch.randn(B, Z),
        "done": torch.zeros(B, dtype=torch.bool),
    }


def test_dqn_victim_aware_forward_shapes():
    B, Z, K = 4, 6, 5
    gen = torch.Generator().manual_seed(111)
    params = init_victim_aware_params(gen, Z, hidden_dim=7, latent_dim=4, K=K)
    q, aux = victim_aware_forward(params, torch.randn(B, Z), torch.randint(0, K, (B,)), K)
    assert q.shape == (B, K)
    assert aux["victim_action_logits"].shape == (B, K)
    assert aux["victim_greedy_logits"].shape == (B, K)
    assert aux["compliance_logit"].shape == (B,)
    assert aux["q_gap"].shape == (B,)
    assert aux["market_aux"].shape == (B, 3)


def test_dqn_victim_aware_train_step_updates_shared_encoder():
    B, Z, K = 8, 6, 5
    gen = torch.Generator().manual_seed(112)
    params = init_victim_aware_params(gen, Z, hidden_dim=7, latent_dim=4, K=K)
    target = clone_params(params)
    before_encoder = {k: params[k].detach().clone() for k in ["enc_W1", "enc_b1", "enc_W2", "enc_b2"]}
    optimizer = torch.optim.Adam(list(params.values()), lr=1e-2)
    metrics = dqn_victim_aware_train_step(params, target, _victim_aware_batch(B, Z, K), optimizer, gamma=0.95, K=K)
    assert {
        "dqn_loss",
        "victim_action_loss",
        "victim_greedy_loss",
        "compliance_loss",
        "q_gap_loss",
        "market_aux_loss",
        "total_loss",
        "q_mean",
        "q_max",
    } == set(metrics)
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.allclose(before_encoder[k], params[k]) for k in before_encoder)


def test_dqn_victim_aware_zero_aux_is_dqn_loss_only():
    B, Z, K = 8, 6, 5
    gen = torch.Generator().manual_seed(113)
    params = init_victim_aware_params(gen, Z, hidden_dim=7, latent_dim=4, K=K)
    target = clone_params(params)
    before_heads = {
        k: v.detach().clone()
        for k, v in params.items()
        if k.startswith(("victim_action_", "victim_greedy_", "compliance_", "q_gap_", "market_aux_"))
    }
    optimizer = torch.optim.SGD(list(params.values()), lr=1e-2)
    metrics = dqn_victim_aware_train_step(
        params,
        target,
        _victim_aware_batch(B, Z, K),
        optimizer,
        gamma=0.95,
        K=K,
        victim_action_coef=0.0,
        victim_greedy_coef=0.0,
        compliance_coef=0.0,
        q_gap_coef=0.0,
        market_aux_coef=0.0,
    )
    assert metrics["total_loss"] == pytest.approx(metrics["dqn_loss"])
    assert all(torch.allclose(before_heads[k], params[k]) for k in before_heads)


def test_dqn_regret_train_step_updates_params():
    B, Z, K = 8, 6, 5
    gen = torch.Generator().manual_seed(9)
    q_params = init_dqn_params(gen, Z, hidden_dim=7, K=K)
    target_q_params = clone_params(q_params)
    regret_params = init_regret_params(gen, Z, hidden_dim=7, K=K)
    before_q = {k: v.detach().clone() for k, v in q_params.items()}
    before_regret = {k: v.detach().clone() for k, v in regret_params.items()}
    optimizer = torch.optim.Adam(list(q_params.values()) + list(regret_params.values()), lr=1e-2)
    batch = {
        "obs": torch.randn(B, Z),
        "action": torch.randint(0, K, (B,)),
        "victim_action": torch.randint(0, K, (B,)),
        "reward": torch.randn(B),
        "next_obs": torch.randn(B, Z),
        "done": torch.zeros(B, dtype=torch.bool),
        "cf_profit": torch.randn(B, K),
    }
    metrics = dqn_regret_train_step(
        q_params,
        target_q_params,
        regret_params,
        batch,
        optimizer,
        gamma=0.95,
        regret_coef=0.1,
    )
    assert set(metrics) == {"dqn_loss", "regret_loss", "total_loss", "q_mean", "q_max"}
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.allclose(before_q[k], q_params[k]) for k in q_params)
    assert any(not torch.allclose(before_regret[k], regret_params[k]) for k in regret_params)


def test_online_loop_smoke():
    config = DQNOracleConfig(
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        rollout_steps=5,
        train_every=2,
        eval_every=10,
        eval_steps=20,
        batch_size=8,
        reservoir_dim=6,
        hidden_dim=8,
        replay_capacity=100,
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert np.isfinite(
        train_df.drop(
            columns=[
                "jepa_loss",
                "regret_loss",
                "victim_action_loss",
                "victim_greedy_loss",
                "compliance_loss",
                "q_gap_loss",
                "market_aux_loss",
                "avg_positive_regret",
                "avg_regret_abs",
                "avg_strategy_entropy",
                "avg_value",
                "lola_immediate_value",
                "lola_future_value",
                "lola_total_value",
                "lola_entropy",
                "model_lola_value",
                "model_lola_value_std",
                "model_lola_entropy",
                "model_lola_immediate_value",
                "model_lola_future_value",
                "model_lola_total_value",
                "model_lola_current_victim_entropy",
                "model_lola_future_victim_entropy",
                "rollout_lola_value_mean",
                "rollout_lola_value_std",
                "rollout_lola_entropy",
                "rollout_lola_best_action_price",
                "rollout_lola_first_step_profit",
                "rollout_lola_future_profit",
                "rollout_lola_victim_price_simulated",
                "rollout_lola_oracle_price_simulated",
                "victim_pred_accuracy",
            ]
        )
        .select_dtypes(include=[float, int])
        .to_numpy()
    ).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_dqn_shared_jepa_online_loop_smoke():
    config = DQNOracleConfig(
        oracle_kind="dqn_shared_jepa",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        rollout_steps=5,
        train_every=2,
        eval_every=10,
        eval_steps=20,
        batch_size=8,
        reservoir_dim=6,
        hidden_dim=8,
        replay_capacity=100,
        jepa_latent_dim=4,
        jepa_coef=0.1,
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert {"dqn_loss", "jepa_loss", "total_loss", "avg_price_oracle", "avg_price_victim"}.issubset(train_df.columns)
    assert np.isfinite(
        train_df.drop(
            columns=[
                "regret_loss",
                "victim_action_loss",
                "victim_greedy_loss",
                "compliance_loss",
                "q_gap_loss",
                "market_aux_loss",
                "avg_positive_regret",
                "avg_regret_abs",
                "avg_strategy_entropy",
                "avg_value",
                "lola_immediate_value",
                "lola_future_value",
                "lola_total_value",
                "lola_entropy",
                "model_lola_value",
                "model_lola_value_std",
                "model_lola_entropy",
                "model_lola_immediate_value",
                "model_lola_future_value",
                "model_lola_total_value",
                "model_lola_current_victim_entropy",
                "model_lola_future_victim_entropy",
                "rollout_lola_value_mean",
                "rollout_lola_value_std",
                "rollout_lola_entropy",
                "rollout_lola_best_action_price",
                "rollout_lola_first_step_profit",
                "rollout_lola_future_profit",
                "rollout_lola_victim_price_simulated",
                "rollout_lola_oracle_price_simulated",
                "victim_pred_accuracy",
            ]
        )
        .select_dtypes(include=[float, int])
        .to_numpy()
    ).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_dqn_regret_online_loop_smoke():
    config = DQNOracleConfig(
        oracle_kind="dqn_regret",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        rollout_steps=5,
        train_every=2,
        eval_every=10,
        eval_steps=20,
        batch_size=8,
        reservoir_dim=6,
        hidden_dim=8,
        replay_capacity=100,
        regret_coef=0.1,
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert {"dqn_loss", "regret_loss", "total_loss", "avg_price_oracle", "avg_price_victim"}.issubset(train_df.columns)
    assert np.isfinite(
        train_df.drop(
            columns=[
                "jepa_loss",
                "victim_action_loss",
                "victim_greedy_loss",
                "compliance_loss",
                "q_gap_loss",
                "market_aux_loss",
                "avg_positive_regret",
                "avg_regret_abs",
                "avg_strategy_entropy",
                "avg_value",
                "lola_immediate_value",
                "lola_future_value",
                "lola_total_value",
                "lola_entropy",
                "model_lola_value",
                "model_lola_value_std",
                "model_lola_entropy",
                "model_lola_immediate_value",
                "model_lola_future_value",
                "model_lola_total_value",
                "model_lola_current_victim_entropy",
                "model_lola_future_victim_entropy",
                "rollout_lola_value_mean",
                "rollout_lola_value_std",
                "rollout_lola_entropy",
                "rollout_lola_best_action_price",
                "rollout_lola_first_step_profit",
                "rollout_lola_future_profit",
                "rollout_lola_victim_price_simulated",
                "rollout_lola_oracle_price_simulated",
                "victim_pred_accuracy",
            ]
        )
        .select_dtypes(include=[float, int])
        .to_numpy()
    ).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_tabular_cfr_online_loop_smoke():
    config = DQNOracleConfig(
        oracle_kind="tabular_cfr",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        eval_every=10,
        eval_steps=20,
        cfr_state_mode="joint_last_action",
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert {"avg_positive_regret", "avg_strategy_entropy", "avg_price_oracle", "avg_price_victim"}.issubset(train_df.columns)
    numeric = train_df.drop(
        columns=[
            "dqn_loss",
            "jepa_loss",
            "regret_loss",
                "victim_action_loss",
                "victim_greedy_loss",
                "compliance_loss",
                "q_gap_loss",
                "market_aux_loss",
            "q_mean",
            "q_max",
            "lola_immediate_value",
            "lola_future_value",
            "lola_total_value",
            "lola_entropy",
            "model_lola_value",
            "model_lola_value_std",
            "model_lola_entropy",
            "model_lola_immediate_value",
            "model_lola_future_value",
            "model_lola_total_value",
            "model_lola_current_victim_entropy",
            "model_lola_future_victim_entropy",
            "rollout_lola_value_mean",
            "rollout_lola_value_std",
            "rollout_lola_entropy",
            "rollout_lola_best_action_price",
            "rollout_lola_first_step_profit",
            "rollout_lola_future_profit",
            "rollout_lola_victim_price_simulated",
            "rollout_lola_oracle_price_simulated",
            "victim_pred_accuracy",
        ]
    ).select_dtypes(include=[float, int]).to_numpy()
    assert np.isfinite(numeric).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_tabular_multi_cfr_loop_smoke():
    config = DQNOracleConfig(
        oracle_kind="tabular_multi_cfr",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        eval_every=10,
        eval_steps=20,
        cfr_state_mode="joint_last_action",
        cfr_value_lr=0.1,
        cfr_gamma=0.95,
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert {"avg_positive_regret", "avg_strategy_entropy", "avg_value", "avg_price_oracle", "avg_price_victim"}.issubset(train_df.columns)
    numeric = train_df.drop(
        columns=[
            "dqn_loss",
            "jepa_loss",
            "regret_loss",
                "victim_action_loss",
                "victim_greedy_loss",
                "compliance_loss",
                "q_gap_loss",
                "market_aux_loss",
            "q_mean",
            "q_max",
            "lola_immediate_value",
            "lola_future_value",
            "lola_total_value",
            "lola_entropy",
            "model_lola_value",
            "model_lola_value_std",
            "model_lola_entropy",
            "model_lola_immediate_value",
            "model_lola_future_value",
            "model_lola_total_value",
            "model_lola_current_victim_entropy",
            "model_lola_future_victim_entropy",
            "rollout_lola_value_mean",
            "rollout_lola_value_std",
            "rollout_lola_entropy",
            "rollout_lola_best_action_price",
            "rollout_lola_first_step_profit",
            "rollout_lola_future_profit",
            "rollout_lola_victim_price_simulated",
            "rollout_lola_oracle_price_simulated",
            "victim_pred_accuracy",
        ]
    ).select_dtypes(include=[float, int]).to_numpy()
    assert np.isfinite(numeric).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_tabular_lola_loop_smoke():
    config = DQNOracleConfig(
        oracle_kind="tabular_lola",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        eval_every=10,
        eval_steps=20,
        lola_gamma=0.95,
        lola_tau=0.05,
        lola_epsilon=0.05,
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert {"lola_immediate_value", "lola_future_value", "lola_total_value", "lola_entropy", "victim_pred_accuracy"}.issubset(train_df.columns)
    numeric = train_df.drop(
        columns=[
            "dqn_loss",
            "jepa_loss",
            "regret_loss",
                "victim_action_loss",
                "victim_greedy_loss",
                "compliance_loss",
                "q_gap_loss",
                "market_aux_loss",
            "q_mean",
            "q_max",
            "avg_positive_regret",
            "avg_regret_abs",
            "avg_strategy_entropy",
            "avg_value",
            "model_lola_value",
            "model_lola_value_std",
            "model_lola_entropy",
            "model_lola_immediate_value",
            "model_lola_future_value",
            "model_lola_total_value",
            "model_lola_current_victim_entropy",
            "model_lola_future_victim_entropy",
            "rollout_lola_value_mean",
            "rollout_lola_value_std",
            "rollout_lola_entropy",
            "rollout_lola_best_action_price",
            "rollout_lola_first_step_profit",
            "rollout_lola_future_profit",
            "rollout_lola_victim_price_simulated",
            "rollout_lola_oracle_price_simulated",
        ]
    ).select_dtypes(include=[float, int]).to_numpy()
    assert np.isfinite(numeric).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_tabular_model_lola_loop_smoke():
    config = DQNOracleConfig(
        oracle_kind="tabular_model_lola",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        eval_every=10,
        eval_steps=20,
        model_lola_gamma=0.95,
        model_lola_tau=0.05,
        model_lola_epsilon=0.02,
        model_lola_victim_policy="epsilon_greedy",
        model_lola_future_policy="epsilon_greedy",
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert {
        "model_lola_value",
        "model_lola_entropy",
        "model_lola_immediate_value",
        "model_lola_future_value",
        "model_lola_current_victim_entropy",
        "model_lola_future_victim_entropy",
        "victim_pred_accuracy",
    }.issubset(train_df.columns)
    numeric = train_df.drop(
        columns=[
            "dqn_loss",
            "jepa_loss",
            "regret_loss",
                "victim_action_loss",
                "victim_greedy_loss",
                "compliance_loss",
                "q_gap_loss",
                "market_aux_loss",
            "q_mean",
            "q_max",
            "avg_positive_regret",
            "avg_regret_abs",
            "avg_strategy_entropy",
            "avg_value",
            "lola_immediate_value",
            "lola_future_value",
            "lola_total_value",
            "lola_entropy",
            "rollout_lola_value_mean",
            "rollout_lola_value_std",
            "rollout_lola_entropy",
            "rollout_lola_best_action_price",
            "rollout_lola_first_step_profit",
            "rollout_lola_future_profit",
            "rollout_lola_victim_price_simulated",
            "rollout_lola_oracle_price_simulated",
        ]
    ).select_dtypes(include=[float, int]).to_numpy()
    assert np.isfinite(numeric).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_tabular_rollout_lola_loop_smoke():
    config = DQNOracleConfig(
        oracle_kind="tabular_rollout_lola",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        eval_every=10,
        eval_steps=20,
        rollout_lola_horizon=3,
        rollout_lola_num_particles=2,
        rollout_lola_tau=0.05,
        rollout_lola_epsilon=0.02,
        rollout_lola_backend="torch",
    )
    result = run_dqn_oracle_vs_qvictim(config)
    train_df = result["train_metrics"]
    eval_df = result["eval_metrics"]
    assert len(train_df) > 0
    assert len(eval_df) > 0
    assert {
        "rollout_lola_value_mean",
        "rollout_lola_value_std",
        "rollout_lola_entropy",
        "rollout_lola_best_action_price",
        "rollout_lola_first_step_profit",
        "rollout_lola_future_profit",
        "rollout_lola_victim_price_simulated",
        "rollout_lola_oracle_price_simulated",
    }.issubset(train_df.columns)
    numeric = train_df.drop(
        columns=[
            "dqn_loss",
            "jepa_loss",
            "regret_loss",
                "victim_action_loss",
                "victim_greedy_loss",
                "compliance_loss",
                "q_gap_loss",
                "market_aux_loss",
            "q_mean",
            "q_max",
            "avg_positive_regret",
            "avg_regret_abs",
            "avg_strategy_entropy",
            "avg_value",
            "lola_immediate_value",
            "lola_future_value",
            "lola_total_value",
            "lola_entropy",
            "model_lola_value",
            "model_lola_value_std",
            "model_lola_entropy",
            "model_lola_immediate_value",
            "model_lola_future_value",
            "model_lola_total_value",
            "model_lola_current_victim_entropy",
            "model_lola_future_victim_entropy",
        ]
    ).select_dtypes(include=[float, int]).to_numpy()
    assert np.isfinite(numeric).all()
    assert np.isfinite(eval_df.select_dtypes(include=[float, int]).to_numpy()).all()


def test_tabular_rollout_lola_progress_jsonl(tmp_path):
    out_dir = tmp_path / "rollout_progress"
    config = DQNOracleConfig(
        oracle_kind="tabular_rollout_lola",
        seed=0,
        B=3,
        H=4,
        K=5,
        total_steps=6,
        log_every=3,
        eval_every=3,
        eval_steps=4,
        rollout_lola_horizon=2,
        rollout_lola_num_particles=2,
        rollout_lola_backend="torch",
        out_dir=str(out_dir),
    )
    run_dqn_oracle_vs_qvictim(config)
    progress_path = out_dir / "progress.jsonl"
    assert progress_path.exists()
    rows = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 2
    assert rows[-1]["step"] == 6
    assert rows[-1]["rollout_lola_backend"] == "torch"
    assert rows[-1]["device"] == "cpu"
    assert "steps_per_second" in rows[-1]


def test_cli_smoke(tmp_path):
    out_dir = tmp_path / "dqn_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--train-every",
        "2",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--batch-size",
        "8",
        "--reservoir-dim",
        "6",
        "--hidden-dim",
        "8",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert "final_eval_avg_profit_oracle" in summary


def test_static_victim_run_writes_summary_kind(tmp_path):
    out_dir = tmp_path / "static_victim"
    config = DQNOracleConfig(
        oracle_kind="tabular_cfr",
        victim_kind="static_cooperative",
        seed=0,
        B=4,
        H=5,
        K=7,
        total_steps=20,
        eval_every=10,
        eval_steps=20,
        out_dir=str(out_dir),
    )
    result = run_dqn_oracle_vs_qvictim(config)
    summary = json.loads((out_dir / "summary.json").read_text())
    assert result["summary"]["victim_kind"] == "static_cooperative"
    assert summary["victim_kind"] == "static_cooperative"


def test_dqn_shared_jepa_cli_smoke(tmp_path):
    out_dir = tmp_path / "dqn_shared_jepa_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "dqn_shared_jepa",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--train-every",
        "2",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--batch-size",
        "8",
        "--reservoir-dim",
        "6",
        "--hidden-dim",
        "8",
        "--jepa-latent-dim",
        "4",
        "--jepa-coef",
        "0.1",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "dqn_shared_jepa"
    assert "final_eval_avg_price_oracle" in summary


def test_dqn_victim_aware_cli_smoke(tmp_path):
    out_dir = tmp_path / "dqn_victim_aware_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "dqn_victim_aware",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--train-every",
        "2",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--batch-size",
        "8",
        "--reservoir-dim",
        "6",
        "--hidden-dim",
        "8",
        "--jepa-latent-dim",
        "4",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "train_metrics.csv").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "dqn_victim_aware"
    train_df = pd.read_csv(out_dir / "train_metrics.csv")
    assert {"victim_action_loss", "victim_greedy_loss", "compliance_loss", "q_gap_loss", "market_aux_loss"}.issubset(train_df.columns)


def test_dqn_regret_cli_smoke(tmp_path):
    out_dir = tmp_path / "dqn_regret_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "dqn_regret",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--train-every",
        "2",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--batch-size",
        "8",
        "--reservoir-dim",
        "6",
        "--hidden-dim",
        "8",
        "--regret-coef",
        "0.1",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "dqn_regret"
    assert "final_eval_avg_price_oracle" in summary


def test_tabular_cfr_cli_smoke(tmp_path):
    out_dir = tmp_path / "tabular_cfr_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "tabular_cfr",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "tabular_cfr"
    assert "final_eval_avg_price_oracle" in summary


def test_eval_modes_tabular_cfr_cli_smoke(tmp_path):
    out_dir = tmp_path / "eval_modes_smoke" / "tabular_cfr_seed0"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "tabular_cfr",
        "--total-steps",
        "2000",
        "--B",
        "16",
        "--H",
        "8",
        "--K",
        "15",
        "--eval-every",
        "1000",
        "--eval-steps",
        "500",
        "--eval-modes",
        "fresh_adaptive,continuation_adaptive,continuation_frozen_greedy,continuation_frozen_epsilon",
        "--out-dir",
        str(out_dir),
        "--seed",
        "0",
    ]
    subprocess.run(cmd, check=True)
    eval_metrics = (out_dir / "eval_metrics.csv").read_text()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert "eval_continuation_frozen_epsilon_victim_avg_epsilon" in eval_metrics
    assert "final_eval_continuation_frozen_greedy_avg_profit_oracle" in summary


def test_cli_smoke_tabular_multi_cfr(tmp_path):
    out_dir = tmp_path / "tabular_multi_cfr_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "tabular_multi_cfr",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--cfr-state-mode",
        "joint_last_action",
        "--cfr-value-lr",
        "0.1",
        "--cfr-gamma",
        "0.95",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "tabular_multi_cfr"
    assert "final_eval_avg_price_oracle" in summary


def test_cli_smoke_tabular_lola(tmp_path):
    out_dir = tmp_path / "tabular_lola_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "tabular_lola",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--lola-gamma",
        "0.95",
        "--lola-tau",
        "0.05",
        "--lola-epsilon",
        "0.05",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "tabular_lola"
    assert "final_eval_avg_price_oracle" in summary


def test_cli_smoke_tabular_model_lola(tmp_path):
    out_dir = tmp_path / "tabular_model_lola_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "tabular_model_lola",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--model-lola-gamma",
        "0.95",
        "--model-lola-tau",
        "0.05",
        "--model-lola-epsilon",
        "0.02",
        "--model-lola-victim-policy",
        "epsilon_greedy",
        "--model-lola-future-policy",
        "epsilon_greedy",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "tabular_model_lola"
    assert "final_eval_avg_price_oracle" in summary


def test_cli_smoke_tabular_rollout_lola(tmp_path):
    out_dir = tmp_path / "tabular_rollout_lola_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "tabular_rollout_lola",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--rollout-lola-horizon",
        "3",
        "--rollout-lola-num-particles",
        "2",
        "--rollout-lola-tau",
        "0.05",
        "--rollout-lola-epsilon",
        "0.02",
        "--rollout-lola-backend",
        "torch",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "progress.jsonl").exists()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "tabular_rollout_lola"
    assert "final_eval_avg_price_oracle" in summary


def test_cli_smoke_tabular_teacher_oracle(tmp_path):
    out_dir = tmp_path / "tabular_teacher_oracle_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "tabular_teacher_oracle",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--teacher-alpha",
        "0.1",
        "--teacher-gamma",
        "0.95",
        "--teacher-hold-duration",
        "6",
        "--teacher-reset-duration",
        "4",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    train = (out_dir / "train_metrics.csv").read_text()
    eval_metrics = (out_dir / "eval_metrics.csv").read_text()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "tabular_teacher_oracle"
    assert "teacher_option_freq_HOLD_HIGH" in train
    assert "teacher_transition_HOLD_HIGH_to_HARVEST_UNDERCUT" in train
    assert "teacher_profit_by_option_HOLD_HIGH" in train
    assert "teacher_market_price_by_option_HOLD_HIGH" in train
    assert "teacher_victim_compliance" in train
    assert "teacher_victim_q_gap" in train
    assert "teacher_victim_high_action_advantage" in train
    assert "teacher_epsilon" in train
    assert "eval_teacher_option_freq_HOLD_HIGH" in eval_metrics
    assert "final_eval_teacher_option_freq_HOLD_HIGH" in summary
    assert summary["teacher_duration_HOLD_HIGH"] == 6
    assert summary["teacher_duration_RESET_HIGH"] == 4
    assert "teacher_anchor_action" in summary
    assert "teacher_anchor_price" in summary


def test_cli_smoke_scripted_carrot_stick(tmp_path):
    out_dir = tmp_path / "scripted_carrot_stick_cli"
    cmd = [
        sys.executable,
        "-m",
        "experiments.dqn_oracle_vs_qvictim",
        "--oracle-kind",
        "scripted_carrot_stick",
        "--total-steps",
        "20",
        "--B",
        "4",
        "--H",
        "5",
        "--K",
        "7",
        "--eval-every",
        "10",
        "--eval-steps",
        "20",
        "--scripted-compliance-low",
        "0.35",
        "--scripted-compliance-high",
        "0.65",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    assert (out_dir / "train_metrics.csv").exists()
    assert (out_dir / "eval_metrics.csv").exists()
    assert (out_dir / "summary.json").exists()
    train = (out_dir / "train_metrics.csv").read_text()
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["oracle_kind"] == "scripted_carrot_stick"
    assert "teacher_option_freq_HOLD_HIGH" in train
    assert "teacher_transition_HOLD_HIGH_to_HARVEST_UNDERCUT" in train
    assert "final_eval_teacher_option_freq_HOLD_HIGH" in summary
