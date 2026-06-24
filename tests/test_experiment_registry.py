from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from experiments.dqn_oracle_config import QVictimOracleConfig
from experiments.experiment_registry import find_completed_run, normalize_config, register_config
from personal_research.scripts import run_jepa_gate


def test_fingerprint_is_stable_across_json_key_order():
    left = {"oracle_kind": "dqn", "seed": 2, "B": 32, "eval_modes": "fresh_adaptive"}
    right = {"eval_modes": "fresh_adaptive", "B": 32, "seed": 2, "oracle_kind": "dqn"}

    assert register_config(left).run_id == register_config(right).run_id


def test_out_dir_does_not_affect_fingerprint():
    left = {"oracle_kind": "dqn", "seed": 2, "out_dir": "results/first"}
    right = {"oracle_kind": "dqn", "seed": 2, "out_dir": "results/second"}

    assert register_config(left).run_id == register_config(right).run_id


def test_initial_victim_path_does_not_affect_fingerprint_but_hash_does():
    left = {
        "oracle_kind": "dqn",
        "seed": 2,
        "initial_victim_state_path": "/machine-a/victim.npz",
        "initial_victim_state_sha256": "a" * 64,
    }
    same_content = {
        **left,
        "initial_victim_state_path": "D:/machine-b/victim.npz",
    }
    different_content = {
        **left,
        "initial_victim_state_sha256": "b" * 64,
    }

    assert register_config(left).run_id == register_config(same_content).run_id
    assert register_config(left).run_id != register_config(different_content).run_id


def test_unused_initial_victim_defaults_do_not_change_legacy_fingerprint():
    legacy = {"oracle_kind": "dqn", "seed": 2}
    explicit_fresh = {
        **legacy,
        "initial_victim_state_mode": "fresh",
        "initial_victim_state_path": None,
        "initial_victim_state_sha256": None,
    }

    assert register_config(legacy).run_id == register_config(explicit_fresh).run_id


def test_seed_affects_only_run_id():
    seed_zero = register_config({"oracle_kind": "dqn_shared_jepa", "seed": 0})
    seed_one = register_config({"oracle_kind": "dqn_shared_jepa", "seed": 1})

    assert seed_zero.spec_id == seed_one.spec_id
    assert seed_zero.run_id != seed_one.run_id


def test_nested_seed_affects_only_run_id():
    seed_zero = register_config({"q_config": {"seed": 0, "alpha": 0.15}, "seed": 0})
    seed_one = register_config({"q_config": {"seed": 1, "alpha": 0.15}, "seed": 1})

    assert seed_zero.spec_id == seed_one.spec_id
    assert seed_zero.run_id != seed_one.run_id


def test_budget_and_eval_modes_define_distinct_runs():
    base = {
        "oracle_kind": "dqn",
        "seed": 0,
        "total_steps": 20_000,
        "eval_modes": "fresh_adaptive,continuation_adaptive",
    }

    assert register_config(base).run_id != register_config({**base, "total_steps": 50_000}).run_id
    assert register_config(base).run_id != register_config({**base, "eval_modes": "fresh_adaptive"}).run_id


def test_missing_legacy_defaults_are_normalized():
    legacy = {"oracle_kind": "dqn_shared_jepa_qdecoder", "seed": 0}
    current = asdict(QVictimOracleConfig(oracle_kind="dqn_shared_jepa_qdecoder", seed=0))

    assert normalize_config(legacy) == normalize_config(current)
    assert register_config(legacy).run_id == register_config(current).run_id


def _write_completed_run(path: Path, config: dict) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "summary.json").write_text("{}", encoding="utf-8")


def test_exact_duplicate_is_found_but_changed_coefficient_is_not(tmp_path):
    config = {"oracle_kind": "dqn_shared_jepa_qdecoder", "seed": 3, "q_decoder_coef": 0.1}
    completed = tmp_path / "study" / "cell" / "seed_3"
    _write_completed_run(completed, config)

    found = find_completed_run({**config, "out_dir": "elsewhere"}, tmp_path)

    assert found is not None
    assert found.path == completed
    assert find_completed_run({**config, "q_decoder_coef": 0.2}, tmp_path) is None


def test_completed_run_prefers_more_artifacts(tmp_path):
    config = {"oracle_kind": "dqn", "seed": 4}
    sparse = tmp_path / "a_sparse"
    rich = tmp_path / "z_rich"
    _write_completed_run(sparse, config)
    _write_completed_run(rich, config)
    (rich / "eval_metrics.csv").write_text("step,value\n1,2\n", encoding="utf-8")

    found = find_completed_run(config, tmp_path)

    assert found is not None
    assert found.path == rich


def _gate_args(root: Path, results_root: Path, *, force: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        root=str(root),
        results_root=str(results_root),
        seeds="0",
        total_steps=20,
        B=4,
        H=2,
        K=5,
        eval_every=10,
        eval_steps=5,
        log_every=5,
        resume=True,
        fail_fast=False,
        dry_run=False,
        force=force,
    )


def _write_gate_result(path: Path, config: dict) -> None:
    _write_completed_run(path, config)
    pd.DataFrame([{"step": 10, "avg_profit_oracle": 0.2}]).to_csv(path / "eval_metrics.csv", index=False)


def test_gate_reuses_existing_run(tmp_path, monkeypatch):
    cell = run_jepa_gate.GateCell("dqn", "dqn")
    monkeypatch.setattr(run_jepa_gate, "CELLS", (cell,))
    args = _gate_args(tmp_path / "new_study", tmp_path / "results")
    task = run_jepa_gate.build_tasks(args)[0]
    existing = tmp_path / "results" / "old_study" / "dqn" / "seed_0"
    _write_gate_result(existing, run_jepa_gate.task_config(task, args))
    called = False

    def fail_run(_task):
        nonlocal called
        called = True
        raise AssertionError("run_task must not be called for an exact completed duplicate")

    monkeypatch.setattr(run_jepa_gate, "run_task", fail_run)

    assert run_jepa_gate.run_gate(args) == 0
    manifest = json.loads((Path(args.root) / "study_manifest.json").read_text(encoding="utf-8"))
    assert called is False
    assert manifest["tasks"][0]["status"] == "reused_existing"
    assert manifest["tasks"][0]["existing_out_dir"] == str(existing)


def test_gate_force_runs_duplicate(tmp_path, monkeypatch):
    cell = run_jepa_gate.GateCell("dqn", "dqn")
    monkeypatch.setattr(run_jepa_gate, "CELLS", (cell,))
    args = _gate_args(tmp_path / "new_study", tmp_path / "results", force=True)
    task = run_jepa_gate.build_tasks(args)[0]
    existing = tmp_path / "results" / "old_study" / "dqn" / "seed_0"
    _write_gate_result(existing, run_jepa_gate.task_config(task, args))
    calls: list[Path] = []

    def fake_run(run_task):
        calls.append(run_task.out_dir)
        _write_gate_result(run_task.out_dir, run_jepa_gate.task_config(run_task, args))
        return {**run_jepa_gate.base_record(run_task, "success"), "returncode": 0}

    monkeypatch.setattr(run_jepa_gate, "run_task", fake_run)

    assert run_jepa_gate.run_gate(args) == 0
    manifest = json.loads((Path(args.root) / "study_manifest.json").read_text(encoding="utf-8"))
    assert calls == [task.out_dir]
    assert manifest["tasks"][0]["status"] == "success"
