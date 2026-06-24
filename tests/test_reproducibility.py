from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from experiments.experiment_registry import register_config
from experiments.reproducibility import collect_reproducibility_metadata, write_metadata


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _init_repo(path: Path) -> None:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.email", "tests@example.com")
    _git(path, "config", "user.name", "Test User")
    (path / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-m", "initial")


def _collect(repo: Path, config: dict | None = None) -> dict:
    return collect_reproducibility_metadata(
        config or {"oracle_kind": "dqn", "seed": 3, "device": "cpu"},
        repo_root=repo,
        started_at="2026-06-14T20:00:00+03:00",
        completed_at=datetime(2026, 6, 14, 18, 30, tzinfo=timezone.utc),
        study_id="representation_gate_20k",
        hypothesis_id="H-REP-001",
    )


def test_collects_clean_repository_runtime_and_fingerprints(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    config = {"oracle_kind": "dqn", "seed": 3, "device": "cpu"}

    metadata = _collect(repo, config)
    registered = register_config(config)

    assert metadata["git_commit"] == _git(repo, "rev-parse", "HEAD")
    assert metadata["dirty_worktree"] is False
    assert metadata["dirty_diff_hash"] is None
    assert metadata["python_version"]
    assert metadata["torch_version"]
    assert metadata["device"] == "cpu"
    assert metadata["hostname"]
    assert metadata["started_at"] == "2026-06-14T17:00:00Z"
    assert metadata["completed_at"] == "2026-06-14T18:30:00Z"
    assert metadata["study_id"] == "representation_gate_20k"
    assert metadata["hypothesis_id"] == "H-REP-001"
    assert metadata["spec_id"] == registered.spec_id
    assert metadata["run_id"] == registered.run_id


def test_collection_has_no_filesystem_write_side_effect(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    before = sorted(path.relative_to(repo) for path in repo.rglob("*"))

    _collect(repo)

    assert sorted(path.relative_to(repo) for path in repo.rglob("*")) == before


def test_dirty_hash_changes_with_tracked_and_untracked_content(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    first = _collect(repo)

    (repo / "untracked.txt").write_text("first\n", encoding="utf-8")
    second = _collect(repo)
    (repo / "untracked.txt").write_text("second\n", encoding="utf-8")
    third = _collect(repo)

    assert first["dirty_worktree"] is True
    assert len(first["dirty_diff_hash"]) == 64
    assert len({first["dirty_diff_hash"], second["dirty_diff_hash"], third["dirty_diff_hash"]}) == 3


def test_rejects_naive_timestamp(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)

    with pytest.raises(ValueError, match="timezone"):
        collect_reproducibility_metadata(
            {"oracle_kind": "dqn", "seed": 0},
            repo_root=repo,
            started_at="2026-06-14T20:00:00",
        )


def test_write_metadata_writes_only_metadata_json(tmp_path):
    metadata = {"run_id": "abc", "completed_at": None}

    path = write_metadata(metadata, tmp_path / "run")

    assert path == tmp_path / "run" / "metadata.json"
    assert json.loads(path.read_text(encoding="utf-8")) == metadata
    assert [item.name for item in path.parent.iterdir()] == ["metadata.json"]
