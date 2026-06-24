"""Compute a log-bin approximation to training-trajectory welfare.

Each progress record contains prices and profits averaged over the preceding
``log_every`` training interval and over vectorized rollout environments.
Consumer surplus is evaluated at that interval's mean price pair. Because the
within-bin price distribution is unavailable, this is a discrete approximation
to mean trajectory welfare, not exact period-level welfare.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calvano_market import CalvanoMarketConfig, market_arrays


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260624

CELLS = (
    ("DQN", "main_public_observation", "results/eval_mode_controls_1m/dqn/seed_*"),
    (
        "Tabular full-information regret matching",
        "main_known_payoff_model",
        "results/eval_mode_controls_1m/tabular_cfr/seed_*",
    ),
    (
        "Latent-prediction DQN (JEPA-style)",
        "main_public_observation",
        "results/pc_new_architectures_100k/shared_jepa/seed_*",
    ),
    (
        "Q-supervised opponent-model DQN",
        "q_supervised_diagnostic",
        "results/pc_new_architectures_100k/victim_aware_dqn/seed_*",
    ),
    (
        "Q-row reconstruction DQN",
        "q_supervised_diagnostic",
        "results/pc_new_architectures_100k/qdecoder_normalized_q/seed_*",
    ),
    (
        "Model-based Q-update rollout planner, H=5",
        "victim_model_access_diagnostic",
        "results/long_matrix_100k_plus/block4_rollout_lola_150k/horizon_5/seed_*",
    ),
)

REQUIRED_PROGRESS_FIELDS = {
    "step",
    "total_steps",
    "avg_price_oracle",
    "avg_price_victim",
    "avg_profit_oracle",
    "avg_profit_victim",
}


def consumer_surplus(price_oracle: float, price_victim: float) -> float:
    config = CalvanoMarketConfig()
    qualities, _ = market_arrays(config)
    utilities = np.array(
        [
            config.outside_quality / config.mu,
            (qualities[0] - price_oracle) / config.mu,
            (qualities[1] - price_victim) / config.mu,
        ],
        dtype=np.float64,
    )
    max_utility = float(np.max(utilities))
    log_sum_exp = max_utility + float(np.log(np.exp(utilities - max_utility).sum()))
    return float(config.demand_scale * config.mu * log_sum_exp)


def load_progress(run_dir: Path) -> pd.DataFrame:
    rows = []
    with (run_dir / "progress.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "log" not in str(row.get("event", "")):
                continue
            missing = REQUIRED_PROGRESS_FIELDS - row.keys()
            if missing:
                raise ValueError(f"{run_dir}: missing progress fields {sorted(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"{run_dir}: no log progress records")
    frame = pd.DataFrame(rows).sort_values("step").drop_duplicates("step", keep="last")
    frame["bin_start"] = frame["step"].shift(fill_value=0).astype(int)
    frame["bin_end"] = frame["step"].astype(int)
    if not (frame["bin_end"] > frame["bin_start"]).all():
        raise ValueError(f"{run_dir}: non-positive or overlapping log bins")
    frame["bin_steps"] = frame["bin_end"] - frame["bin_start"]
    frame["consumer_surplus_approx"] = [
        consumer_surplus(price_o, price_v)
        for price_o, price_v in zip(
            frame["avg_price_oracle"],
            frame["avg_price_victim"],
            strict=True,
        )
    ]
    frame["joint_profit"] = frame["avg_profit_oracle"] + frame["avg_profit_victim"]
    frame["total_welfare_approx"] = frame["consumer_surplus_approx"] + frame["joint_profit"]
    return frame


def weighted_window(frame: pd.DataFrame, window_steps: int | None) -> tuple[pd.DataFrame, dict[str, float]]:
    total_steps = int(frame["total_steps"].iloc[-1])
    window_start = 0 if window_steps is None else max(0, total_steps - window_steps)
    selected = frame.copy()
    selected["overlap_steps"] = (
        np.minimum(selected["bin_end"], total_steps)
        - np.maximum(selected["bin_start"], window_start)
    ).clip(lower=0)
    selected = selected[selected["overlap_steps"] > 0].copy()
    covered_steps = int(selected["overlap_steps"].sum())
    expected_steps = total_steps - window_start
    if covered_steps != expected_steps:
        raise ValueError(
            f"incomplete log coverage: expected {expected_steps} steps, found {covered_steps}"
        )

    weights = selected["overlap_steps"].to_numpy(dtype=np.float64)

    def average(column: str) -> float:
        return float(np.average(selected[column].to_numpy(dtype=np.float64), weights=weights))

    metrics = {
        "window_start_step": window_start,
        "window_end_step": total_steps,
        "covered_steps": covered_steps,
        "N_bins": len(selected),
        "min_bin_steps": int(selected["bin_steps"].min()),
        "max_bin_steps": int(selected["bin_steps"].max()),
        "avg_price_oracle": average("avg_price_oracle"),
        "avg_price_victim": average("avg_price_victim"),
        "avg_profit_oracle": average("avg_profit_oracle"),
        "avg_profit_victim": average("avg_profit_victim"),
        "joint_profit": average("joint_profit"),
        "consumer_surplus_approx": average("consumer_surplus_approx"),
        "total_welfare_approx": average("total_welfare_approx"),
    }
    return selected, metrics


def mature_reference() -> pd.DataFrame:
    rows = []
    for seed in range(10):
        directory = (
            ROOT / "results" / "mature_q_vs_q_checkpoint_10m"
            if seed == 0
            else ROOT / "results" / f"mature_q_vs_q_checkpoint_10m_seed_{seed}"
        )
        final = json.loads((directory / "summary.json").read_text(encoding="utf-8"))["final_eval"]
        cs = consumer_surplus(float(final["price_agent_0"]), float(final["price_agent_1"]))
        joint_profit = float(final["profit_agent_0"]) + float(final["profit_agent_1"])
        rows.append(
            {
                "cell": "Mature Q-vs-Q final policy",
                "layer": "reference",
                "window": "final_policy_evaluation",
                "seed": seed,
                "avg_price_oracle": float(final["price_agent_0"]),
                "avg_price_victim": float(final["price_agent_1"]),
                "avg_profit_oracle": float(final["profit_agent_0"]),
                "avg_profit_victim": float(final["profit_agent_1"]),
                "joint_profit": joint_profit,
                "consumer_surplus_approx": cs,
                "total_welfare_approx": cs + joint_profit,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_difference(
    values: np.ndarray,
    reference: np.ndarray,
    draws: int = BOOTSTRAP_DRAWS,
) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    value_means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    reference_means = reference[
        rng.integers(0, len(reference), size=(draws, len(reference)))
    ].mean(axis=1)
    low, high = np.quantile(value_means - reference_means, [0.025, 0.975])
    return float(low), float(high)


def aggregate(seed_level: pd.DataFrame, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = (
        "consumer_surplus_approx",
        "joint_profit",
        "total_welfare_approx",
        "avg_price_oracle",
        "avg_price_victim",
    )
    aggregate_rows = []
    contrast_rows = []
    for (cell, layer, window), group in seed_level.groupby(["cell", "layer", "window"], sort=False):
        aggregate_row: dict[str, object] = {
            "cell": cell,
            "layer": layer,
            "window": window,
            "N_seed": int(group["seed"].nunique()),
            "window_steps": int(group["covered_steps"].iloc[0]),
            "min_bins_per_seed": int(group["N_bins"].min()),
            "max_bins_per_seed": int(group["N_bins"].max()),
            "min_log_bin_steps": int(group["min_bin_steps"].min()),
            "max_log_bin_steps": int(group["max_bin_steps"].max()),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=np.float64)
            aggregate_row[f"{metric}_mean"] = float(values.mean())
            aggregate_row[f"{metric}_sample_sd"] = float(values.std(ddof=1))
        aggregate_rows.append(aggregate_row)

        if window != "last_100k":
            continue
        for metric in ("consumer_surplus_approx", "joint_profit", "total_welfare_approx"):
            values = group[metric].to_numpy(dtype=np.float64)
            reference_values = reference[metric].to_numpy(dtype=np.float64)
            low, high = bootstrap_difference(values, reference_values)
            contrast_rows.append(
                {
                    "cell": cell,
                    "layer": layer,
                    "window": window,
                    "metric": metric,
                    "N_seed": len(values),
                    "reference_N_seed": len(reference_values),
                    "mean": float(values.mean()),
                    "reference_mean": float(reference_values.mean()),
                    "difference_vs_mature_q_vs_q": float(values.mean() - reference_values.mean()),
                    "bootstrap_difference_ci_low": low,
                    "bootstrap_difference_ci_high": high,
                }
            )
    return pd.DataFrame(aggregate_rows), pd.DataFrame(contrast_rows)


def write_report(output: Path, contrasts: pd.DataFrame) -> None:
    lines = [
        "# Log-Bin Trajectory Welfare Approximation",
        "",
        "Each progress record is an average over the preceding training log interval",
        "and over vectorized rollout environments. Consumer surplus is evaluated at",
        "the interval mean price pair, then combined with interval mean realized",
        "profits. Intervals are weighted by their number of training steps.",
        "",
        "Because within-bin prices were not saved, this estimates",
        "`CS(E[p | bin])` rather than `E[CS(p) | bin]`. It is a binned trajectory",
        "approximation, not exact period-level trajectory welfare.",
        "",
        "The main window is the last 100,000 training steps. The mature Q-vs-Q",
        "reference is the ten-seed final-policy evaluation distribution, because its",
        "progress file contains checkpoint policy evaluations rather than matching",
        "training-bin averages.",
        "",
        "## Last-100k Contrasts",
        "",
        "| Learner | Metric | Difference vs mature Q-vs-Q | Bootstrap 95% CI |",
        "|---|---|---:|---:|",
    ]
    for row in contrasts.itertuples(index=False):
        lines.append(
            f"| {row.cell} | {row.metric} | "
            f"{row.difference_vs_mature_q_vs_q:+.6f} | "
            f"[{row.bootstrap_difference_ci_low:+.6f}, "
            f"{row.bootstrap_difference_ci_high:+.6f}] |"
        )
    lines.extend(
        [
            "",
            "Bootstrap intervals independently resample the ten learner runs and ten",
            "mature reference checkpoints with 10,000 percentile-bootstrap draws.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="analysis/binned_trajectory_welfare")
    parser.add_argument(
        "--paper-table",
        default="paper_preprint/generated_tables/binned_trajectory_welfare_last_100k.csv",
    )
    args = parser.parse_args()

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    bin_frames = []
    seed_rows = []
    for cell, layer, pattern in CELLS:
        run_dirs = sorted(ROOT.glob(pattern))
        valid_dirs = [
            directory
            for directory in run_dirs
            if (directory / "summary.json").exists()
            and int(json.loads((directory / "config.json").read_text())["total_steps"]) >= 100_000
        ]
        if len(valid_dirs) != 10:
            raise ValueError(f"{cell}: expected 10 completed runs, found {len(valid_dirs)}")
        for run_dir in valid_dirs:
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            seed = int(config["seed"])
            progress = load_progress(run_dir)
            for window_name, window_steps in (("full_training", None), ("last_100k", 100_000)):
                selected, metrics = weighted_window(progress, window_steps)
                selected.insert(0, "cell", cell)
                selected.insert(1, "layer", layer)
                selected.insert(2, "window", window_name)
                selected["seed"] = seed
                selected.insert(3, "result_path", str(run_dir.relative_to(ROOT)))
                bin_frames.append(selected)
                seed_rows.append(
                    {
                        "cell": cell,
                        "layer": layer,
                        "window": window_name,
                        "seed": seed,
                        "result_path": str(run_dir.relative_to(ROOT)),
                        **metrics,
                    }
                )

    bins = pd.concat(bin_frames, ignore_index=True)
    seed_level = pd.DataFrame(seed_rows)
    reference = mature_reference()
    aggregate_frame, contrasts = aggregate(seed_level, reference)

    bins.to_csv(output / "bin_level.csv", index=False)
    seed_level.to_csv(output / "seed_level.csv", index=False)
    reference.to_csv(output / "mature_q_vs_q_reference.csv", index=False)
    aggregate_frame.to_csv(output / "aggregate.csv", index=False)
    contrasts.to_csv(output / "contrasts_vs_mature_q_vs_q.csv", index=False)
    write_report(output, contrasts)

    paper_table = ROOT / args.paper_table
    paper_table.parent.mkdir(parents=True, exist_ok=True)
    contrasts.to_csv(paper_table, index=False)
    print(contrasts.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
