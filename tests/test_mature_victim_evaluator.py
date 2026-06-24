from __future__ import annotations

import json

import pandas as pd

from scripts.evaluate_mature_victim_gate import evaluate


def _write_task(root, cell, seed, profit_10k, profit_20k, market, **train):
    out = root / cell / f"seed_{seed}"
    out.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "step": 10_000,
                "eval_continuation_adaptive_avg_profit_oracle": profit_10k,
                "eval_continuation_adaptive_market_price_mean": market,
            },
            {
                "step": 20_000,
                "eval_continuation_adaptive_avg_profit_oracle": profit_20k,
                "eval_continuation_adaptive_market_price_mean": market,
            },
        ]
    ).to_csv(out / "eval_metrics.csv", index=False)
    if train:
        pd.DataFrame([train, train]).to_csv(out / "train_metrics.csv", index=False)
    return {
        "cell": cell,
        "seed": seed,
        "out_dir": str(out),
        "status": "success",
    }


def test_paired_evaluator_admits_stable_improvement_and_rejects_diagnostic(tmp_path):
    tasks = []
    for seed in range(3):
        tasks.append(_write_task(tmp_path, "dqn_control", seed, 0.30, 0.30, 1.65))
        tasks.append(
            _write_task(
                tmp_path,
                "imitation_option_dqn",
                seed,
                0.307,
                0.308,
                1.66,
            )
        )
        tasks.append(
            _write_task(
                tmp_path,
                "imitation_bc_frozen",
                seed,
                0.32,
                0.32,
                1.80,
            )
        )
    (tmp_path / "study_manifest.json").write_text(
        json.dumps({"tasks": tasks}),
        encoding="utf-8",
    )

    _, decisions = evaluate(tmp_path, tmp_path / "analysis")
    by_cell = decisions.set_index("cell")

    assert by_cell.loc["imitation_option_dqn", "decision"] == "ADMIT"
    assert by_cell.loc["imitation_bc_frozen", "decision"] == "REJECT"
    assert bool(by_cell.loc["imitation_bc_frozen", "diagnostic_only"]) is True
