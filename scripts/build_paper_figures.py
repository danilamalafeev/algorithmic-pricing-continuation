"""Build paper figures from existing analysis artifacts.

This script is intentionally read-only with respect to experiment outputs. It
derives a small manuscript figure package from CSV/Markdown analysis files and
writes the generated artifacts under ``paper_figures/``.
"""

from __future__ import annotations

import math
import json
import re
from pathlib import Path

import pandas as pd

from calvano_market import CalvanoMarketConfig, build_static_benchmarks, market_arrays


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
OUT = ROOT / "paper_figures"
Q_INERTIA = ANALYSIS / "q_inertia_confirmatory_final_victim_20k"


DISPLAY_NAMES = {
    "dqn": "DQN",
    "rollout_lola": "Rollout LOLA",
    "shared_jepa": "Shared-JEPA",
    "shared_jepa_qdecoder": "Q-decoder + variance",
    "shared_jepa_qdecoder_no_collapse": "Q-decoder no penalty",
    "victim_aware_dqn": "Victim-aware DQN",
}

WELFARE_TRAJECTORY_SPECS = [
    (
        "3M Victim 1M",
        "imitation_option_dqn",
        [0, 1, 2, 5, 6, 7],
        ROOT / "results" / "mature_victim_3m_1m_followup",
    ),
    (
        "3M Victim 1M",
        "dqn_control",
        [0, 1, 2, 5, 6, 7],
        ROOT / "results" / "mature_victim_3m_1m_followup",
    ),
    (
        "10M Victim 1M trajectory subset",
        "imitation_option_dqn",
        [0, 1, 2],
        ROOT / "results" / "mature_victim_matched_10m_1m",
    ),
    (
        "10M Victim 1M trajectory subset",
        "dqn_control",
        [0, 1, 2],
        ROOT / "results" / "mature_victim_matched_10m_1m",
    ),
]


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
        }
    )
    return plt


def _read_q_threshold() -> tuple[float, float]:
    path = ANALYSIS / "mature_q_vs_q_checkpoint_10m" / "checkpoint_summary.csv"
    row = pd.read_csv(path).iloc[0]
    return float(row["final_symmetric_profit"]), float(row["final_price"])


def _read_scripted_harvest_reference() -> float:
    """Extract the highest scripted-harvest reference from RESEARCH_OVERVIEW.md."""
    text = (ROOT / "RESEARCH_OVERVIEW.md").read_text()
    match = re.search(r"monopoly tick 2\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([+-][0-9.]+)", text)
    if not match:
        raise RuntimeError("Could not find scripted harvest reference in RESEARCH_OVERVIEW.md")
    return float(match.group(2))


def build_scripted_harvest_reference() -> pd.DataFrame:
    """Materialize the hand-coded scripted-harvest reference table.

    This is not learned performance. The values are parsed from the project
    overview and exported so the paper figure package has a machine-readable
    provenance row instead of depending on a manuscript table.
    """
    text = (ROOT / "RESEARCH_OVERVIEW.md").read_text()
    pattern = re.compile(
        r"(?P<anchor>q_vs_q|monopoly) tick (?P<tick>\d+)\s+"
        r"(?P<seeds>\d+)\s+"
        r"(?P<continuation_oracle_profit>[0-9.]+)\s+"
        r"(?P<frozen_oracle_profit>[0-9.]+)\s+"
        r"(?P<market_price>[0-9.]+)\s+"
        r"(?P<oracle_victim_profit_gap>[+-][0-9.]+)"
    )
    rows = []
    for match in pattern.finditer(text):
        row = match.groupdict()
        row["cell"] = f"{row['anchor']} tick {row['tick']}"
        row["seed_count"] = int(row.pop("seeds"))
        row["tick"] = int(row["tick"])
        for key in ["continuation_oracle_profit", "frozen_oracle_profit", "market_price", "oracle_victim_profit_gap"]:
            row[key] = float(row[key])
        row["source"] = "RESEARCH_OVERVIEW.md"
        row["source_type"] = "hand_coded_scripted_reference_not_learned"
        rows.append(row)
    if not rows:
        raise RuntimeError("Could not parse scripted harvest reference table from RESEARCH_OVERVIEW.md")
    out = pd.DataFrame(rows)
    out["is_best_scripted_reference"] = out["continuation_oracle_profit"].eq(out["continuation_oracle_profit"].max())
    return out.sort_values(["anchor", "tick"])


def _welfare_inputs() -> tuple[CalvanoMarketConfig, pd.Series, pd.Series]:
    config = CalvanoMarketConfig(m=15)
    qualities, costs = market_arrays(config)
    return config, pd.Series(qualities), pd.Series(costs)


def _consumer_surplus(prices: pd.DataFrame, config: CalvanoMarketConfig, qualities: pd.Series) -> pd.Series:
    utilities = prices.sub(qualities.to_numpy(), axis=1).mul(-1.0 / config.mu)
    outside = config.outside_quality / config.mu
    max_u = pd.concat([utilities.max(axis=1), pd.Series(outside, index=utilities.index)], axis=1).max(axis=1)
    exp_inside = (utilities.sub(max_u, axis=0)).map(lambda value: math.exp(float(value)))
    exp_outside = (max_u.mul(-1.0).add(outside)).map(lambda value: math.exp(float(value)))
    inclusive_value = (exp_inside.sum(axis=1) + exp_outside).map(lambda value: math.log(float(value)))
    return config.demand_scale * config.mu * (max_u + inclusive_value)


def _static_welfare(prices: pd.DataFrame) -> pd.DataFrame:
    config, qualities, costs = _welfare_inputs()
    shares_util = prices.sub(qualities.to_numpy(), axis=1).mul(-1.0 / config.mu)
    max_u = pd.concat(
        [shares_util.max(axis=1), pd.Series(config.outside_quality / config.mu, index=shares_util.index)],
        axis=1,
    ).max(axis=1)
    exp_inside = (shares_util.sub(max_u, axis=0)).map(lambda value: math.exp(float(value)))
    exp_outside = (max_u.mul(-1.0).add(config.outside_quality / config.mu)).map(
        lambda value: math.exp(float(value))
    )
    denom = exp_inside.sum(axis=1) + exp_outside
    demand = exp_inside.div(denom, axis=0).mul(config.demand_scale)
    profits = prices.sub(costs.to_numpy(), axis=1).mul(demand)
    out = pd.DataFrame(index=prices.index)
    out["consumer_surplus"] = _consumer_surplus(prices, config, qualities)
    out["oracle_profit"] = profits.iloc[:, 0]
    out["victim_profit"] = profits.iloc[:, 1]
    out["joint_profit"] = profits.sum(axis=1)
    out["total_welfare"] = out["consumer_surplus"] + out["joint_profit"]
    out["outside_share"] = exp_outside / denom
    return out


def build_welfare_deterministic_action_pairs(q_threshold: float, q_market_price: float) -> pd.DataFrame:
    config = CalvanoMarketConfig(m=15)
    benchmarks = build_static_benchmarks(config)
    action_rows = [
        ("Discrete Nash pair", int(benchmarks.nash_actions[0]), int(benchmarks.nash_actions[1]), "static_grid_pair"),
        ("One-tick undercut mechanism", 7, 8, "static_grid_pair"),
        ("Symmetric Q-vs-Q final price pair", 9, 9, "static_grid_pair"),
        ("Discrete monopoly pair", int(benchmarks.monopoly_actions[0]), int(benchmarks.monopoly_actions[1]), "static_grid_pair"),
    ]
    rows = []
    for label, oracle_action, victim_action, source_type in action_rows:
        prices = pd.DataFrame(
            [[benchmarks.price_grid[oracle_action], benchmarks.price_grid[victim_action]]],
            columns=["oracle_price", "victim_price"],
        )
        welfare = _static_welfare(prices).iloc[0].to_dict()
        rows.append(
            {
                "label": label,
                "source_type": source_type,
                "oracle_action": oracle_action,
                "victim_action": victim_action,
                "oracle_price": float(prices["oracle_price"].iloc[0]),
                "victim_price": float(prices["victim_price"].iloc[0]),
                "market_price": float(prices.mean(axis=1).iloc[0]),
                **welfare,
            }
        )
    df = pd.DataFrame(rows)
    q_row = df[df["label"].eq("Symmetric Q-vs-Q final price pair")].iloc[0]
    undercut_row = df[df["label"].eq("One-tick undercut mechanism")].iloc[0]
    df["consumer_surplus_minus_q_vs_q_price_pair"] = df["consumer_surplus"] - float(q_row["consumer_surplus"])
    df["total_welfare_minus_q_vs_q_price_pair"] = df["total_welfare"] - float(q_row["total_welfare"])
    df["consumer_surplus_minus_one_tick_undercut"] = df["consumer_surplus"] - float(undercut_row["consumer_surplus"])
    df["total_welfare_minus_one_tick_undercut"] = df["total_welfare"] - float(undercut_row["total_welfare"])
    df["q_vs_q_checkpoint_profit_reference"] = q_threshold
    df["q_vs_q_checkpoint_market_price"] = q_market_price
    return df


def _trajectory_welfare(path: Path) -> pd.Series:
    data = pd.read_csv(path)
    prices = data[["oracle_price", "victim_price"]].astype(float)
    welfare = _static_welfare(prices)
    return pd.Series(
        {
            "rows": len(data),
            "oracle_profit_mean": float(data["oracle_profit"].mean()),
            "victim_profit_mean": float(data["victim_profit"].mean()),
            "joint_profit_mean": float((data["oracle_profit"] + data["victim_profit"]).mean()),
            "market_price_mean": float(data["market_price"].mean()),
            "consumer_surplus_mean": float(welfare["consumer_surplus"].mean()),
            "total_welfare_mean": float(welfare["total_welfare"].mean()),
            "outside_share_mean": float(welfare["outside_share"].mean()),
            "oracle_price_mean": float(data["oracle_price"].mean()),
            "victim_price_mean": float(data["victim_price"].mean()),
        }
    )


def build_welfare_trajectory_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    for study, cell, seeds, root in WELFARE_TRAJECTORY_SPECS:
        for seed in seeds:
            path = root / cell / f"seed_{seed}" / "trajectory_diagnostics" / "final_eval_continuation_adaptive.csv"
            if not path.exists():
                seed_rows.append(
                    {
                        "study": study,
                        "cell": cell,
                        "seed": seed,
                        "trajectory_present": False,
                        "source": str(path.relative_to(ROOT)),
                    }
                )
                continue
            row = _trajectory_welfare(path).to_dict()
            row.update(
                {
                    "study": study,
                    "cell": cell,
                    "seed": seed,
                    "trajectory_present": True,
                    "source": str(path.relative_to(ROOT)),
                }
            )
            seed_rows.append(row)
    seed_df = pd.DataFrame(seed_rows)
    present = seed_df[seed_df["trajectory_present"].eq(True)].copy()
    agg_rows = []
    metrics = [
        "oracle_profit_mean",
        "victim_profit_mean",
        "joint_profit_mean",
        "market_price_mean",
        "consumer_surplus_mean",
        "total_welfare_mean",
        "outside_share_mean",
        "oracle_price_mean",
        "victim_price_mean",
    ]
    for (study, cell), group in present.groupby(["study", "cell"], sort=False):
        row = {
            "study": study,
            "cell": cell,
            "seed_count": int(group["seed"].nunique()),
            "row_count": int(group["rows"].sum()),
            "source_type": "trajectory_diagnostics_final_eval_continuation_adaptive",
        }
        for metric in metrics:
            row[metric] = float(group[metric].mean())
            row[f"{metric}_sample_sd"] = float(group[metric].std())
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    for study, group in agg.groupby("study", sort=False):
        controls = group[group["cell"].eq("dqn_control")]
        if controls.empty:
            continue
        control = controls.iloc[0]
        for idx in group.index:
            for metric in ["consumer_surplus_mean", "total_welfare_mean", "joint_profit_mean", "market_price_mean"]:
                agg.loc[idx, f"{metric}_minus_dqn"] = float(agg.loc[idx, metric]) - float(control[metric])
    return seed_df, agg


def build_welfare_architecture_probe_summary(q_market_price: float) -> pd.DataFrame:
    """Approximate welfare for 100k architecture probes from final aggregate prices.

    This uses per-seed final evaluation average prices and realized mean profits,
    not full trajectory rows. It is therefore an aggregate-price welfare summary,
    separate from the trajectory-backed mature-Victim welfare table.
    """
    cells = [
        "imitation_option_dqn",
        "shared_jepa",
        "victim_aware_dqn",
        "qdecoder_normalized_q",
        "qdecoder_centered_advantages",
        "qdecoder_normalized_q_delta",
    ]
    labels = {
        "imitation_option_dqn": "Imitation-option DQN",
        "shared_jepa": "Shared-JEPA",
        "victim_aware_dqn": "Victim-aware DQN",
        "qdecoder_normalized_q": "Q-decoder Q",
        "qdecoder_centered_advantages": "Q-decoder advantages",
        "qdecoder_normalized_q_delta": "Q-decoder delta",
    }
    q_prices = pd.DataFrame([[q_market_price, q_market_price]], columns=["oracle_price", "victim_price"])
    q_welfare = _static_welfare(q_prices).iloc[0]
    rows = []
    for cell in cells:
        for seed in [0, 1, 2]:
            path = ROOT / "results" / "pc_new_architectures_100k" / cell / f"seed_{seed}" / "eval_metrics.csv"
            if not path.exists():
                continue
            final = pd.read_csv(path).query("step == 100000").iloc[0]
            prices = pd.DataFrame(
                [
                    [
                        float(final["eval_continuation_adaptive_avg_price_oracle"]),
                        float(final["eval_continuation_adaptive_avg_price_victim"]),
                    ]
                ],
                columns=["oracle_price", "victim_price"],
            )
            welfare = _static_welfare(prices).iloc[0]
            joint_profit = float(final["eval_continuation_adaptive_avg_profit_oracle"]) + float(
                final["eval_continuation_adaptive_avg_profit_victim"]
            )
            rows.append(
                {
                    "cell": cell,
                    "label": labels[cell],
                    "seed": seed,
                    "source_type": "final_eval_aggregate_prices_not_full_trajectory",
                    "oracle_profit": float(final["eval_continuation_adaptive_avg_profit_oracle"]),
                    "victim_profit": float(final["eval_continuation_adaptive_avg_profit_victim"]),
                    "joint_profit": joint_profit,
                    "oracle_price": float(prices["oracle_price"].iloc[0]),
                    "victim_price": float(prices["victim_price"].iloc[0]),
                    "market_price": float(final["eval_continuation_adaptive_market_price_mean"]),
                    "consumer_surplus": float(welfare["consumer_surplus"]),
                    "total_welfare": float(welfare["consumer_surplus"] + joint_profit),
                    "consumer_surplus_minus_q_vs_q_price_pair": float(
                        welfare["consumer_surplus"] - q_welfare["consumer_surplus"]
                    ),
                    "total_welfare_minus_q_vs_q_price_pair": float(
                        welfare["consumer_surplus"] + joint_profit - q_welfare["total_welfare"]
                    ),
                    "market_price_minus_q_vs_q": float(final["eval_continuation_adaptive_market_price_mean"])
                    - q_market_price,
                }
            )
    seed_df = pd.DataFrame(rows)
    metrics = [
        "oracle_profit",
        "victim_profit",
        "joint_profit",
        "oracle_price",
        "victim_price",
        "market_price",
        "consumer_surplus",
        "total_welfare",
        "consumer_surplus_minus_q_vs_q_price_pair",
        "total_welfare_minus_q_vs_q_price_pair",
        "market_price_minus_q_vs_q",
    ]
    summary_rows = []
    for (cell, label), group in seed_df.groupby(["cell", "label"], sort=False):
        row = {
            "cell": cell,
            "label": label,
            "seed_count": int(group["seed"].nunique()),
            "source_type": "final_eval_aggregate_prices_not_full_trajectory",
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sample_sd"] = float(group[metric].std())
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def build_main_result_data(q_threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    mode_controls = pd.read_csv(ROOT / "results" / "eval_mode_controls_1m" / "aggregate_by_mode.csv")
    long_labels = {
        "dqn": "1M DQN continuation",
        "tabular_cfr": "1M tabular CFR continuation",
    }
    for _, row in mode_controls.iterrows():
        if row["mode"] not in long_labels:
            continue
        rows.append(
            {
                "label": long_labels.get(str(row["mode"]), str(row["mode"])),
                "group": "1M continuation controls",
                "profit_mean": row["final_eval_continuation_adaptive_avg_profit_oracle_mean"],
                "profit_std": row["final_eval_continuation_adaptive_avg_profit_oracle_std"],
                "market_price_mean": row["final_eval_continuation_adaptive_market_price_mean_mean"],
                "seed_count": 3,
                "std_type": "sample_sd_across_seed_final_means",
                "evaluation_mode": "continuation_adaptive",
                "final_step": 1_000_000,
                "is_rounded_reference": False,
                "source": "results/eval_mode_controls_1m/aggregate_by_mode.csv",
            }
        )

    pc = pd.read_csv(ANALYSIS / "pc_new_architectures_100k" / "economic_aggregate.csv")
    pc_final = pc[pc["step"] == 100000]
    pc_labels = {
        "imitation_option_dqn": "100k imitation-option",
        "shared_jepa": "100k Shared-JEPA",
        "victim_aware_dqn": "100k Victim-aware DQN",
        "qdecoder_normalized_q": "100k Q-decoder Q",
        "qdecoder_centered_advantages": "100k Q-decoder adv",
        "qdecoder_normalized_q_delta": "100k Q-decoder delta",
    }
    for _, row in pc_final.iterrows():
        rows.append(
            {
                "label": pc_labels.get(str(row["cell"]), str(row["cell"])),
                "group": "100k PC architecture matrix",
                "profit_mean": row["oracle_profit_mean"],
                "profit_std": row["oracle_profit_std"],
                "market_price_mean": row["market_price_mean"],
                "seed_count": 3,
                "std_type": "sample_sd_across_seed_final_means",
                "evaluation_mode": "continuation_adaptive",
                "final_step": 100_000,
                "is_rounded_reference": False,
                "source": "analysis/pc_new_architectures_100k/economic_aggregate.csv",
            }
        )

    def matched_rows(study: str, label_prefix: str) -> list[dict[str, object]]:
        path = ANALYSIS / study / "paired_checkpoint_metrics.csv"
        rows = pd.read_csv(path)
        final = rows[rows["step"] == 1_000_000]
        seed_count = int(final["seed"].nunique())
        return [
            {
                "label": f"{label_prefix} DQN",
                "group": "1M matched gate",
                "profit_mean": final["control_continuation_adaptive_oracle_profit"].mean(),
                "profit_std": final["control_continuation_adaptive_oracle_profit"].std(),
                "market_price_mean": final["control_continuation_adaptive_market_price"].mean(),
                "seed_count": seed_count,
                "std_type": "sample_sd_across_seed_final_means",
                "evaluation_mode": "continuation_adaptive",
                "final_step": 1_000_000,
                "is_rounded_reference": False,
                "source": f"analysis/{study}/paired_checkpoint_metrics.csv",
            },
            {
                "label": f"{label_prefix} imitation",
                "group": "1M matched gate",
                "profit_mean": final["continuation_adaptive_oracle_profit"].mean(),
                "profit_std": final["continuation_adaptive_oracle_profit"].std(),
                "market_price_mean": final["continuation_adaptive_market_price"].mean(),
                "seed_count": seed_count,
                "std_type": "sample_sd_across_seed_final_means",
                "evaluation_mode": "continuation_adaptive",
                "final_step": 1_000_000,
                "is_rounded_reference": False,
                "source": f"analysis/{study}/paired_checkpoint_metrics.csv",
            },
        ]

    rows.extend(matched_rows("mature_victim_matched_10m_1m", "10M Victim"))
    rows.extend(matched_rows("mature_victim_3m_1m_followup", "3M Victim"))
    rows.append(
        {
            "label": "Scripted harvest reference",
            "group": "hand-coded reference",
            "profit_mean": _read_scripted_harvest_reference(),
            "profit_std": pd.NA,
            "market_price_mean": pd.NA,
            "seed_count": pd.NA,
            "std_type": "rounded_hand_coded_reference",
            "evaluation_mode": "hand_coded_reference_not_learned",
            "final_step": pd.NA,
            "is_rounded_reference": True,
            "source": "paper_figures/scripted_harvest_reference.csv",
        }
    )
    df = pd.DataFrame(rows)
    df["minus_q_vs_q"] = df["profit_mean"] - q_threshold
    return df


def plot_main_results(df: pd.DataFrame, q_threshold: float) -> None:
    plt = _setup_matplotlib()
    order = [
        "100k Q-decoder delta",
        "100k Q-decoder adv",
        "100k Q-decoder Q",
        "100k Victim-aware DQN",
        "100k Shared-JEPA",
        "100k imitation-option",
        "1M tabular CFR continuation",
        "1M DQN continuation",
        "10M Victim DQN",
        "3M Victim DQN",
        "3M Victim imitation",
        "10M Victim imitation",
        "Scripted harvest reference",
    ]
    plot_df = df.set_index("label").loc[order].reset_index()
    colors = plot_df["group"].map(
        {
            "100k PC architecture matrix": "#8a8f98",
            "1M continuation controls": "#756bb1",
            "1M matched gate": "#377eb8",
            "hand-coded reference": "#e07a2f",
        }
    )
    colors = colors.mask(plot_df["label"].str.contains("imitation"), "#2ca25f")

    fig, ax = plt.subplots(figsize=(8.3, 5.6))
    y = list(range(len(plot_df)))
    ax.scatter(plot_df["profit_mean"], y, c=colors, s=58, zorder=3)
    err_df = plot_df[pd.to_numeric(plot_df["profit_std"], errors="coerce").notna()]
    err_y = [plot_df.index[plot_df["label"].eq(label)][0] for label in err_df["label"]]
    ax.errorbar(
        err_df["profit_mean"],
        err_y,
        xerr=err_df["profit_std"].astype(float),
        fmt="none",
        ecolor="#333333",
        elinewidth=0.9,
        capsize=2.5,
        zorder=2,
    )
    ax.axvline(q_threshold, color="#b2182b", linestyle="--", linewidth=1.4)
    ax.text(q_threshold + 0.003, 0.15, "Q-vs-Q 10M threshold", color="#b2182b", fontsize=9)
    ax.set_yticks(y, plot_df["label"])
    ax.set_xlabel("Oracle continuation-adaptive profit")
    ax.set_title("Dynamic-continuation outcomes vs Q-vs-Q benchmark")
    ax.set_xlim(0.0, 0.39)
    fig.tight_layout()
    fig.savefig(OUT / "main_result_dot.png")
    fig.savefig(OUT / "main_result_bar.png")
    plt.close(fig)


def build_maturity_sweep(q_threshold: float) -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS / "mature_victim_maturity_sweep_100k" / "admission_by_age.csv")
    order = ["fresh", "100k", "1m", "3m", "10m"]
    df["age"] = pd.Categorical(df["age"], categories=order, ordered=True)
    df = df.sort_values("age")
    df["q_vs_q_threshold"] = q_threshold
    return df


def build_maturity_seed_data() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS / "mature_victim_maturity_sweep_100k" / "paired_by_age_seed.csv")
    order = ["fresh", "100k", "1m", "3m", "10m"]
    df["age"] = pd.Categorical(df["age"], categories=order, ordered=True)
    return df.sort_values(["age", "seed"])


def plot_maturity_sweep(df: pd.DataFrame, seed_df: pd.DataFrame, q_threshold: float) -> None:
    plt = _setup_matplotlib()
    x = list(range(len(df)))
    age_to_x = {str(age): i for i, age in enumerate(df["age"].astype(str))}
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for _, row in seed_df.iterrows():
        xpos = age_to_x[str(row["age"])]
        ax.scatter(xpos - 0.08, row["oracle_profit"], color="#2ca25f", alpha=0.45, s=22)
        ax.scatter(xpos + 0.08, row["control_oracle_profit"], color="#377eb8", alpha=0.45, s=22)
    ax.plot(x, df["imitation_mean_profit"], marker="o", linewidth=2, color="#2ca25f", label="Imitation-option mean")
    ax.plot(x, df["dqn_mean_profit"], marker="o", linewidth=2, color="#377eb8", label="DQN control mean")
    ax.axhline(q_threshold, color="#b2182b", linestyle="--", linewidth=1.2, label="Q-vs-Q threshold")
    ax.set_xticks(x, df["age"].astype(str))
    ax.set_ylabel("Continuation-adaptive Oracle profit")
    ax.set_xlabel("Victim checkpoint age")
    ax.set_title("100k maturity sweep: seed markers and mean lines")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "maturity_sweep_100k_seeded.png")
    fig.savefig(OUT / "maturity_sweep_100k.png")
    plt.close(fig)


def build_mechanism_data() -> pd.DataFrame:
    rows = []
    ten_m = pd.read_csv(ANALYSIS / "mechanism_audit_10m_1m" / "trajectory_mechanism_summary.csv")
    ten_m = ten_m[ten_m["cell"] == "imitation_option_dqn"]
    for _, row in ten_m.iterrows():
        rows.append(
            {
                "study": "10M Victim 1M",
                "scope": f"seed {int(row['seed'])}",
                "rows": int(row["rows"]),
                "dominant_option_frequency": float(row["dominant_option_freq"]),
                "dominant_oracle_action": int(row["dominant_oracle_action"]),
                "mean_price_gap": float(row["price_gap_victim_minus_oracle_mean"]),
                "oracle_action_entropy": float(row["oracle_action_entropy"]),
                "victim_action_entropy": float(row["victim_action_entropy"]),
                "source": "analysis/mechanism_audit_10m_1m/trajectory_mechanism_summary.csv",
            }
        )
    three_m = pd.read_csv(ANALYSIS / "mature_victim_3m_1m_followup" / "mechanism_summary.csv").iloc[0]
    rows.append(
        {
            "study": "3M Victim 1M",
            "scope": str(three_m["scope"]),
            "rows": int(three_m["trajectory_rows"]),
            "dominant_option_frequency": float(three_m["dominant_option_frequency"]),
            "dominant_oracle_action": int(three_m["dominant_oracle_action"]),
            "mean_price_gap": float(three_m["mean_price_gap_victim_minus_oracle"]),
            "oracle_action_entropy": float(three_m["oracle_action_entropy_bits"]),
            "victim_action_entropy": float(three_m["victim_action_entropy_bits"]),
            "source": "analysis/mature_victim_3m_1m_followup/mechanism_summary.csv",
        }
    )
    return pd.DataFrame(rows)


def plot_mechanism(df: pd.DataFrame) -> None:
    plt = _setup_matplotlib()
    metric_specs = [
        ("dominant_option_frequency", "HARVEST_UNDERCUT_1 frequency", (0.0, 1.05), "#2ca25f"),
        ("mean_price_gap", "Victim - Oracle price gap", (0.0, 0.045), "#377eb8"),
        ("oracle_action_entropy", "Oracle action entropy (bits)", (0.0, 0.07), "#756bb1"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0), sharex=False)
    groups = [
        ("10M per-seed trajectory audit", df[df["study"].eq("10M Victim 1M")]),
        ("3M aggregate trajectory summary", df[df["study"].eq("3M Victim 1M")]),
    ]
    for row_idx, (row_title, group) in enumerate(groups):
        labels = group["scope"].astype(str)
        for col_idx, (metric, title, ylim, color) in enumerate(metric_specs):
            ax = axes[row_idx][col_idx]
            ax.bar(labels, group[metric], color=color, alpha=0.9)
            ax.set_title(title if row_idx == 0 else "")
            ax.set_ylim(*ylim)
            if col_idx == 0:
                ax.set_ylabel(row_title)
            ax.tick_params(axis="x", labelrotation=25)
    fig.suptitle("Mechanism audit by denominator: one-tick undercut maintenance", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "mechanism_by_denominator.png", bbox_inches="tight")
    fig.savefig(OUT / "mechanism_one_tick.png", bbox_inches="tight")
    plt.close(fig)


def _mean_summary(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["seed_count"] = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_median"] = float(group[metric].median())
            row[f"{metric}_sample_sd"] = float(group[metric].std()) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_q_inertia_summaries() -> dict[str, pd.DataFrame]:
    """Build manuscript-facing summaries from confirmatory Q-inertia CSVs.

    The source rows are one statistical seed. Vectorized environments are
    already aggregated inside each seed row and are not treated as observations.
    """
    source_root = Q_INERTIA.relative_to(ROOT).as_posix()
    q = pd.read_csv(Q_INERTIA / "q_table_inertia_audit.csv")
    frozen = pd.read_csv(Q_INERTIA / "frozen_vs_adaptive.csv")
    eps = pd.read_csv(Q_INERTIA / "epsilon_floor_stress.csv")
    alpha = pd.read_csv(Q_INERTIA / "alpha_sensitivity.csv")
    primitive = pd.read_csv(Q_INERTIA / "primitive_policy_tests.csv")
    availability = pd.read_csv(Q_INERTIA / "input_availability.csv")

    key_metrics = [
        "eval_avg_profit_oracle",
        "eval_avg_profit_victim",
        "eval_joint_profit",
        "eval_market_price_mean",
        "eval_victim_avg_epsilon",
        "dominant_joint_action_freq",
        "q_max_abs_change",
        "q_mean_abs_change_visited_state_actions",
        "greedy_switches_all_states_mean_per_vector_env",
        "dominant_state_q_gap_mean",
        "dominant_state_greedy_switch_rate",
        "consumer_surplus_mean",
        "total_welfare_mean",
    ]
    q_summary = _mean_summary(q, ["age", "cell"], key_metrics)
    q_summary["source"] = f"{source_root}/q_table_inertia_audit.csv"
    q_summary["interpretation_denominator"] = "mechanism_diagnostic_not_headline_gate"

    adaptive_index = q.set_index(["age", "cell", "seed"]).sort_index()
    frozen_index = frozen.set_index(["age", "cell", "seed"]).sort_index()
    diff_rows = []
    diff_metrics = [
        "eval_avg_profit_oracle",
        "eval_avg_profit_victim",
        "eval_market_price_mean",
        "dominant_joint_action_freq",
        "dominant_state_q_gap_mean",
        "q_max_abs_change",
        "greedy_switches_all_states_mean_per_vector_env",
    ]
    for age, cell in sorted(set((idx[0], idx[1]) for idx in adaptive_index.index)):
        adaptive_seeds = set(adaptive_index.loc[(age, cell)].index)
        frozen_seeds = set(frozen_index.loc[(age, cell)].index)
        seeds = sorted(adaptive_seeds & frozen_seeds)
        row = {"age": age, "cell": cell, "paired_seed_count": len(seeds)}
        for metric in diff_metrics:
            diffs = adaptive_index.loc[(age, cell, seeds), metric].to_numpy() - frozen_index.loc[
                (age, cell, seeds), metric
            ].to_numpy()
            row[f"{metric}_adaptive_minus_frozen_mean"] = float(diffs.mean()) if len(diffs) else pd.NA
            row[f"{metric}_adaptive_minus_frozen_max_abs"] = float(abs(diffs).max()) if len(diffs) else pd.NA
        diff_rows.append(row)
    adaptive_frozen = pd.DataFrame(diff_rows)
    adaptive_frozen["source_adaptive"] = f"{source_root}/q_table_inertia_audit.csv"
    adaptive_frozen["source_frozen"] = f"{source_root}/frozen_vs_adaptive.csv"

    eps_summary = _mean_summary(
        eps,
        ["age", "cell", "epsilon_floor"],
        [
            "eval_avg_profit_oracle",
            "eval_market_price_mean",
            "dominant_joint_action_freq",
            "greedy_switches_all_states_mean_per_vector_env",
            "dominant_state_q_gap_mean",
            "consumer_surplus_mean",
            "total_welfare_mean",
        ],
    )
    eps_summary["source"] = f"{source_root}/epsilon_floor_stress.csv"

    alpha_summary = _mean_summary(
        alpha,
        ["age", "cell", "alpha_multiplier"],
        [
            "eval_avg_profit_oracle",
            "eval_market_price_mean",
            "dominant_joint_action_freq",
            "greedy_switches_all_states_mean_per_vector_env",
            "dominant_state_q_gap_mean",
        ],
    )
    alpha_summary["source"] = f"{source_root}/alpha_sensitivity.csv"

    primitive_summary = _mean_summary(
        primitive,
        ["age", "policy"],
        [
            "eval_avg_profit_oracle",
            "eval_market_price_mean",
            "dominant_joint_action_freq",
            "greedy_switches_all_states_mean_per_vector_env",
            "dominant_state_q_gap_mean",
            "consumer_surplus_mean",
            "total_welfare_mean",
        ],
    )
    primitive_summary["source"] = f"{source_root}/primitive_policy_tests.csv"

    available_summary = (
        availability.groupby(["age", "cell"], dropna=False)["available"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "available_seed_runs", "count": "planned_seed_runs"})
    )
    available_summary["missing_seed_runs"] = (
        available_summary["planned_seed_runs"] - available_summary["available_seed_runs"]
    )

    return {
        "q_inertia_summary": q_summary,
        "q_inertia_adaptive_minus_frozen": adaptive_frozen,
        "q_inertia_epsilon_floor_summary": eps_summary,
        "q_inertia_alpha_sensitivity_summary": alpha_summary,
        "q_inertia_primitive_policy_summary": primitive_summary,
        "q_inertia_input_availability": available_summary,
    }


def plot_q_inertia_summary(q_summary: pd.DataFrame) -> None:
    plt = _setup_matplotlib()
    plot_df = q_summary.copy()
    plot_df["label"] = plot_df["age"] + " " + plot_df["cell"].map(
        {"imitation_option_dqn": "imitation", "dqn_control": "DQN"}
    )
    plot_df = plot_df.sort_values(["age", "cell"])
    x = range(len(plot_df))

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9))
    axes[0].bar(x, plot_df["dominant_state_q_gap_mean_mean"], color="#377eb8")
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_title("Anchor Q-gap")
    axes[0].set_ylabel("Q-gap at dominant state")

    axes[1].bar(x, plot_df["greedy_switches_all_states_mean_per_vector_env_mean"], color="#756bb1")
    axes[1].set_title("Greedy switches")
    axes[1].set_ylabel("Mean switches per vector env")

    axes[2].bar(x, plot_df["dominant_joint_action_freq_mean"], color="#2ca25f")
    axes[2].set_title("Dominant action-pair frequency")
    axes[2].set_ylabel("Frequency")
    axes[2].set_ylim(0, 1.0)

    for ax in axes:
        ax.set_xticks(list(x), plot_df["label"], rotation=25, ha="right")
    fig.suptitle("Confirmatory Q-table inertia diagnostics", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "q_inertia_anchor_gap.png", bbox_inches="tight")
    plt.close(fig)


def plot_q_inertia_epsilon_stress(eps_summary: pd.DataFrame, q_threshold: float) -> None:
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    floors = ["baseline", "0.01", "0.05", "0.1"]
    floor_labels = ["baseline", "0.01", "0.05", "0.10"]
    for ax, age in zip(axes, ["3M", "10M"]):
        group = eps_summary[(eps_summary["age"].eq(age)) & (eps_summary["cell"].eq("imitation_option_dqn"))].copy()
        group["epsilon_floor"] = group["epsilon_floor"].astype(str)
        values = []
        freqs = []
        for floor in floors:
            row = group[group["epsilon_floor"].eq(floor)]
            values.append(float(row["eval_avg_profit_oracle_mean"].iloc[0]))
            freqs.append(float(row["dominant_joint_action_freq_mean"].iloc[0]))
        ax.plot(floor_labels, values, marker="o", color="#2ca25f", linewidth=2, label="Oracle profit")
        ax.axhline(q_threshold, color="#b2182b", linestyle="--", linewidth=1.1, label="Q-vs-Q profit")
        for xpos, freq in enumerate(freqs):
            ax.text(xpos, values[xpos] + 0.004, f"pair={freq:.2f}", ha="center", fontsize=8)
        ax.set_title(f"{age} imitation under epsilon floors")
        ax.set_xlabel("Evaluation epsilon floor")
        ax.set_ylim(0.24, 0.335)
    axes[0].set_ylabel("Oracle profit")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "q_inertia_epsilon_floor_stress.png", bbox_inches="tight")
    plt.close(fig)


def plot_q_inertia_primitive_policy(primitive_summary: pd.DataFrame, q_threshold: float) -> None:
    plt = _setup_matplotlib()
    order = [
        "PURE_UNDERCUT_1",
        "PURE_MATCH",
        "PURE_OVERCUT_1",
        "PURE_RESET_HIGH",
        "PURE_UNDERCUT_2",
        "RANDOM_OPTION",
    ]
    labels = {
        "PURE_UNDERCUT_1": "undercut 1",
        "PURE_MATCH": "match",
        "PURE_OVERCUT_1": "overcut 1",
        "PURE_RESET_HIGH": "reset high",
        "PURE_UNDERCUT_2": "undercut 2",
        "RANDOM_OPTION": "random",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharey=True)
    for ax, age in zip(axes, ["3M", "10M"]):
        group = primitive_summary[primitive_summary["age"].eq(age)].set_index("policy").loc[order].reset_index()
        colors = ["#2ca25f" if policy == "PURE_UNDERCUT_1" else "#8a8f98" for policy in group["policy"]]
        ax.bar(range(len(group)), group["eval_avg_profit_oracle_mean"], color=colors)
        ax.axhline(q_threshold, color="#b2182b", linestyle="--", linewidth=1.1)
        ax.set_xticks(range(len(group)), [labels[p] for p in group["policy"]], rotation=30, ha="right")
        ax.set_title(f"{age} evaluation-only primitive policies")
        ax.set_xlabel("Primitive policy")
    axes[0].set_ylabel("Oracle profit")
    fig.tight_layout()
    fig.savefig(OUT / "q_inertia_primitive_policy.png", bbox_inches="tight")
    plt.close(fig)


def build_provenance_data() -> pd.DataFrame:
    evals = pd.read_csv(ANALYSIS / "mechanism_audit_10m_1m" / "eval_consistency.csv")
    counts = (
        evals.groupby(["cell", "status"], dropna=False)
        .size()
        .reset_index(name="run_count")
        .sort_values(["cell", "status"])
    )
    return counts


def build_provenance_detail() -> pd.DataFrame:
    evals = pd.read_csv(ANALYSIS / "mechanism_audit_10m_1m" / "eval_consistency.csv")
    keep = [
        "cell",
        "seed",
        "status",
        "spec_id",
        "run_id",
        "duplicate_run_id",
        "config_total_steps",
        "config_initial_victim_sha",
        "has_final_oracle_state",
        "has_final_victim_state",
        "eval_rows",
        "has_required_checkpoints",
        "matches_analysis_profit",
        "eval_final_profit",
        "eval_final_market_price",
    ]
    return evals[keep].sort_values(["cell", "seed"])


def build_representation_seed_data() -> pd.DataFrame:
    rep = pd.read_csv(ANALYSIS / "representation_gate_20k" / "target_nonstationarity.csv")
    final = rep[
        (rep["record_type"] == "eval_point")
        & (rep["step"] == 20000)
        & (rep["eval_mode"] == "continuation_adaptive")
    ].copy()
    final["label"] = final["cell"].map(DISPLAY_NAMES).fillna(final["cell"])
    keep = [
        "label",
        "cell",
        "seed",
        "step",
        "eval_mode",
        "eval_avg_profit_oracle",
        "eval_avg_profit_victim",
        "eval_market_price_mean",
        "eval_profit_asymmetry",
        "market_regime",
        "market_position_nash_to_monopoly",
    ]
    return final[keep].sort_values(["label", "seed"])


def build_exact_margin_audit(q_threshold: float) -> pd.DataFrame:
    rows = []
    studies = [
        ("10M Victim 1M", "mature_victim_matched_10m_1m", "admission_summary.csv"),
        ("3M Victim 1M", "mature_victim_3m_1m_followup", "admission_summary.csv"),
    ]
    for label, study, summary_name in studies:
        paired = pd.read_csv(ANALYSIS / study / "paired_checkpoint_metrics.csv")
        final = paired[paired["step"] == 1_000_000]
        summary = pd.read_csv(ANALYSIS / study / summary_name).iloc[0]
        mean_profit = float(final["continuation_adaptive_oracle_profit"].mean())
        row = {
            "study": label,
            "exact_q_vs_q_threshold": q_threshold,
            "mean_imitation_profit_from_paired_csv": mean_profit,
            "exact_margin_from_paired_csv": mean_profit - q_threshold,
            "source_paired_csv": f"analysis/{study}/paired_checkpoint_metrics.csv",
            "source_admission_summary": f"analysis/{study}/{summary_name}",
        }
        if "q_vs_q_symmetric_profit_threshold" in summary:
            rounded_threshold = float(summary["q_vs_q_symmetric_profit_threshold"])
            row["admission_summary_threshold"] = rounded_threshold
            row["margin_using_admission_summary_threshold"] = mean_profit - rounded_threshold
            row["threshold_rounding_delta"] = rounded_threshold - q_threshold
        else:
            row["admission_summary_threshold"] = pd.NA
            row["margin_using_admission_summary_threshold"] = pd.NA
            row["threshold_rounding_delta"] = pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def plot_provenance(df: pd.DataFrame) -> None:
    plt = _setup_matplotlib()
    pivot = df.pivot(index="cell", columns="status", values="run_count").fillna(0)
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    bottom = None
    colors = {"reused_existing": "#8a8f98", "success": "#2ca25f"}
    for status in pivot.columns:
        ax.bar(pivot.index, pivot[status], bottom=bottom, label=status, color=colors.get(status, None))
        bottom = pivot[status] if bottom is None else bottom + pivot[status]
    ax.set_ylabel("Runs in raw audit")
    ax.set_title("10M-1M raw audit provenance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "provenance_reuse_status.png")
    plt.close(fig)


def write_captions(q_threshold: float, q_market_price: float) -> None:
    captions = f"""# Paper Figure Captions

Generated by:

```bash
python scripts/build_paper_figures.py
```

## Figure 1: Main Dynamic-Continuation Results

File: `main_result_dot.png`

Dots show Oracle `continuation_adaptive` profit for learned controls and matched gates. Whiskers are seed standard deviations where available, not confidence intervals. The dashed red line is the mature 10M Q-vs-Q benchmark, profit `{q_threshold:.10f}` and market price `{q_market_price:.10f}`. Main negative controls are computed from `results/eval_mode_controls_1m/aggregate_by_mode.csv` and `analysis/pc_new_architectures_100k/economic_aggregate.csv`; the 10M/3M matched gates are computed from their `paired_checkpoint_metrics.csv` files. The scripted-harvest reference is hand-coded, rounded, sourced from `paper_figures/scripted_harvest_reference.csv` parsed from `RESEARCH_OVERVIEW.md`, and visually separated as a non-learned reference. The plot deliberately uses a zero-based x-axis to avoid visually exaggerating small margins. The 20k representation gate is kept as a target-diagnostic appendix artifact, not as main architecture evidence.

## Figure 2: 100k Victim-Maturity Sweep

File: `maturity_sweep_100k_seeded.png`

The line plot shows mean imitation-option and DQN control profits across Victim checkpoint ages at the 100k Oracle-training budget, with individual seed markers overlaid. Imitation has higher mean Oracle profit than DQN across ages, but the seed-level paired deltas are positive in only `2/3` seeds at 100k and 1m. No point crosses the Q-vs-Q threshold. This shows a DQN-relative gap at the 100k budget without crossing the Q-vs-Q threshold; it is not a fresh-creation result.

## Figure 3: One-Tick Mechanism Audit

File: `mechanism_by_denominator.png`

The plot summarizes the low-entropy one-tick maintenance mechanism while separating denominators. The top row uses 10M per-seed trajectory-audit rows from `analysis/mechanism_audit_10m_1m/trajectory_mechanism_summary.csv`; the bottom row uses the 3M aggregate `all_candidate_seeds` row from `analysis/mature_victim_3m_1m_followup/mechanism_summary.csv`. The denominators differ and should not be pooled.

## Figure 4: 10M-1M Raw-Audit Provenance

File: `provenance_reuse_status.png`

The stacked bars show that the 10M-1M raw audit validates a mixed-provenance bundle: DQN seeds 0-2 and imitation seeds 0-1 are `reused_existing`, while imitation seed 2 is `success`. This belongs in the appendix/provenance disclosure.

## Figure 5: Confirmatory Q-Table Inertia Diagnostics

File: `q_inertia_anchor_gap.png`

The three-panel figure summarizes the final-Victim Q-inertia audit from `analysis/q_inertia_confirmatory_final_victim_20k/q_table_inertia_audit.csv`. It reports anchor Q-gap, greedy-switch counts, and dominant action-pair frequency by Victim age and Oracle cell. The statistical unit is the seed; vectorized environments are within-seed batches. The 20k diagnostic covers all planned learned seed-runs: 10M seeds `0/1/2/3/4` and 3M seeds `0/1/2/5/6/7`.

## Figure 6: Epsilon-Floor Stress

File: `q_inertia_epsilon_floor_stress.png`

The plot shows imitation-option Oracle profit under baseline evaluation and epsilon floors of 0.01, 0.05, and 0.10. The stress test identifies a boundary: injected Victim exploration sharply reduces diagnostic profit and disrupts the dominant one-tick pair frequency. This is evidence for deterministic-path/Q-table inertia, not broad adaptive robustness.

## Figure 7: Primitive-Policy Mechanism Test

File: `q_inertia_primitive_policy.png`

The plot compares evaluation-only primitive policies. `PURE_UNDERCUT_1` closely matches the learned imitation mechanism, while match, overcut, reset-high, two-tick undercut, and random-option controls do not. This supports attributing the mechanism to the designer-specified public-history undercut primitive rather than autonomous discovery.

## Appendix Tables

- `provenance_reuse_detail.csv`: raw-audit `spec_id`, `run_id`, status, checkpoint, and consistency columns.
- `maturity_sweep_100k_seed_data.csv`: seed-level maturity sweep values.
- `representation_gate_20k_seed_data.csv`: seed-level final continuation-adaptive representation-gate values.
- `exact_margin_audit.csv`: exact Q-vs-Q margin calculations using updated 3M and 10M paired CSVs.
- `welfare_deterministic_action_pairs.csv`: logit consumer-surplus and total-welfare checks for Nash, one-tick undercut, Q-vs-Q final-price, and monopoly grid action pairs.
- `welfare_trajectory_summary.csv`: trajectory-backed welfare summaries for 3M 6-seed continuation windows and the local 10M 3-seed trajectory subset. The 10M 5-seed headline profit result remains sourced from the updated PC analysis, not from this trajectory subset.
- `welfare_architecture_probe_summary.csv`: aggregate-price welfare summary for the 100k tested architecture probes. It computes logit consumer surplus from final mean Oracle/Victim prices and adds realized mean firm profits; it is not a full-trajectory welfare table.
- `q_inertia_summary.csv`: seed-level mean summary for Q-table movement, anchor Q-gap, greedy switches, and dominant pair frequency.
- `q_inertia_adaptive_minus_frozen.csv`: matched adaptive-minus-frozen differences for the final-Victim diagnostic subset.
- `q_inertia_epsilon_floor_summary.csv`: epsilon-floor stress summary.
- `q_inertia_alpha_sensitivity_summary.csv`: Victim alpha-multiplier stress summary.
- `q_inertia_primitive_policy_summary.csv`: evaluation-only primitive-policy mechanism tests.
- `q_inertia_input_availability.csv`: planned versus available learned seed-runs for the Q-inertia diagnostic.
- `scripted_harvest_reference.csv`: machine-readable hand-coded scripted-harvest reference parsed from `RESEARCH_OVERVIEW.md`; this is not learned evidence.
"""
    (OUT / "FIGURE_CAPTIONS.md").write_text(captions)


def write_provenance_summary() -> None:
    q = pd.read_csv(ANALYSIS / "mature_q_vs_q_checkpoint_10m" / "checkpoint_summary.csv").iloc[0]
    ten = pd.read_csv(ANALYSIS / "mature_victim_matched_10m_1m" / "admission_summary.csv").iloc[0]
    three = pd.read_csv(ANALYSIS / "mature_victim_3m_1m_followup" / "admission_summary.csv").iloc[0]
    q_manifest = json.loads((Q_INERTIA / "MANIFEST.json").read_text())
    q_manifest_args = q_manifest.get("args", {})
    q_availability = pd.read_csv(Q_INERTIA / "input_availability.csv")
    q_seed_rows = []
    for age in ["10M", "3M"]:
        seeds = sorted(int(x) for x in q_availability.loc[q_availability["age"].eq(age), "seed"].unique())
        q_seed_rows.append(f"{age}: {'/'.join(str(x) for x in seeds)}")
    q_seed_summary = "; ".join(q_seed_rows)
    text = f"""# Provenance Summary

| Item | Value |
|---|---|
| Q-vs-Q checkpoint SHA | `{q['checkpoint_sha256']}` |
| Q-table SHA | `{q['q_table_sha256']}` |
| Q-vs-Q profit | `{float(q['final_symmetric_profit']):.16f}` |
| Q-vs-Q market price | `{float(q['final_price']):.16f}` |
| Q-vs-Q converged | `{q['converged']}` |
| Q-vs-Q completed steps | `{int(q['completed_steps'])}` / `{int(q['total_steps'])}` |
| 10M-1M checkpoint SHA | `{ten['checkpoint_sha256']}` |
| 10M-1M compatibility verified | `{ten['compatibility_verified']}` |
| 3M-1M checkpoint SHA | `{three['checkpoint_sha256']}` |
| 3M-1M compatibility verified | `{three['compatibility_verified']}` |
| Canonical Q-inertia source | `analysis/q_inertia_confirmatory_final_victim_20k` |
| Q-inertia horizon | `{q_manifest_args.get('horizons')}` |
| Q-inertia Victim start | `{q_manifest_args.get('victim_start')}` |
| Q-inertia seed coverage | `{q_seed_summary}` |
| Superseded Q-inertia diagnostics | Earlier 2k Q-inertia diagnostic directories are superseded for manuscript use |

See `provenance_reuse_status.csv`, `provenance_reuse_status.png`, and `provenance_reuse_detail.csv` for the raw-audit reuse disclosure.
"""
    (OUT / "PROVENANCE_SUMMARY.md").write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    q_threshold, q_market_price = _read_q_threshold()

    build_scripted_harvest_reference().to_csv(OUT / "scripted_harvest_reference.csv", index=False)

    main_df = build_main_result_data(q_threshold)
    main_df.to_csv(OUT / "main_result_data.csv", index=False)
    plot_main_results(main_df, q_threshold)

    maturity_df = build_maturity_sweep(q_threshold)
    maturity_df.to_csv(OUT / "maturity_sweep_100k_data.csv", index=False)
    maturity_seed_df = build_maturity_seed_data()
    maturity_seed_df.to_csv(OUT / "maturity_sweep_100k_seed_data.csv", index=False)
    plot_maturity_sweep(maturity_df, maturity_seed_df, q_threshold)

    mechanism_df = build_mechanism_data()
    mechanism_df.to_csv(OUT / "mechanism_one_tick_data.csv", index=False)
    plot_mechanism(mechanism_df)

    provenance_df = build_provenance_data()
    provenance_df.to_csv(OUT / "provenance_reuse_status.csv", index=False)
    plot_provenance(provenance_df)
    build_provenance_detail().to_csv(OUT / "provenance_reuse_detail.csv", index=False)
    build_representation_seed_data().to_csv(OUT / "representation_gate_20k_seed_data.csv", index=False)
    build_exact_margin_audit(q_threshold).to_csv(OUT / "exact_margin_audit.csv", index=False)
    build_welfare_deterministic_action_pairs(q_threshold, q_market_price).to_csv(
        OUT / "welfare_deterministic_action_pairs.csv", index=False
    )
    welfare_seed_df, welfare_summary_df = build_welfare_trajectory_summary()
    welfare_seed_df.to_csv(OUT / "welfare_trajectory_seed_data.csv", index=False)
    welfare_summary_df.to_csv(OUT / "welfare_trajectory_summary.csv", index=False)
    build_welfare_architecture_probe_summary(q_market_price).to_csv(
        OUT / "welfare_architecture_probe_summary.csv", index=False
    )

    q_inertia_tables = build_q_inertia_summaries()
    for name, table in q_inertia_tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
    plot_q_inertia_summary(q_inertia_tables["q_inertia_summary"])
    plot_q_inertia_epsilon_stress(q_inertia_tables["q_inertia_epsilon_floor_summary"], q_threshold)
    plot_q_inertia_primitive_policy(q_inertia_tables["q_inertia_primitive_policy_summary"], q_threshold)

    write_captions(q_threshold, q_market_price)
    write_provenance_summary()
    print(f"Wrote paper figures to {OUT}")


if __name__ == "__main__":
    main()
