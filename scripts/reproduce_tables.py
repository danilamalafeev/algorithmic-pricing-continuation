"""Reproduce manuscript tables from the experiment registry.

The script treats the seed as the statistical unit. Vectorized environments are
rollout batches inside a seed, not independent replications.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from calvano_market import CalvanoMarketConfig, market_arrays, profit_vector


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260623


def _contains(text: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: df["result_path"].astype(str).str.contains(text, na=False)


def _and(*predicates: Callable[[pd.DataFrame], pd.Series]) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: np.logical_and.reduce([pred(df) for pred in predicates])


def _eq(column: str, value: object) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: df[column] == value


CELLS = [
    {
        "study": "P2 1M controls",
        "cell": "DQN",
        "layer": "public_bandit_feedback",
        "predicate": _and(_contains("results/eval_mode_controls_1m/dqn/seed_"), _eq("architecture", "dqn"), _eq("steps", 1_000_000)),
        "train_horizon_steps": "1,000,000",
        "eval_metric_prefix": "final_eval_continuation_adaptive",
        "initial_condition": "Own-run final checkpoint",
        "manuscript_use": "Main control evidence",
        "information_set": "Public runtime state; realized reward feedback",
    },
    {
        "study": "P2 1M controls",
        "cell": "Tabular full-information regret matching",
        "layer": "known_payoff_model",
        "predicate": _and(_contains("results/eval_mode_controls_1m/tabular_cfr/seed_"), _eq("architecture", "tabular_cfr"), _eq("steps", 1_000_000)),
        "train_horizon_steps": "1,000,000",
        "eval_metric_prefix": "final_eval_continuation_adaptive",
        "initial_condition": "Own-run final checkpoint",
        "manuscript_use": "Known-payoff-model control",
        "information_set": "Public runtime state; counterfactual payoffs for all own actions",
    },
    {
        "study": "100k public-observation architecture probes",
        "cell": "Latent-prediction DQN (JEPA-style)",
        "layer": "public_bandit_feedback",
        "predicate": _and(_contains("results/pc_new_architectures_100k/shared_jepa/seed_"), _eq("architecture", "dqn_shared_jepa"), _eq("steps", 100_000)),
        "train_horizon_steps": "100,000",
        "eval_metric_prefix": "final_eval_continuation_adaptive",
        "initial_condition": "Own-run final checkpoint",
        "manuscript_use": "Public-state representation probe",
        "information_set": "Public runtime state; realized reward; next-latent auxiliary target",
    },
    {
        "study": "100k public-observation architecture probes",
        "cell": "Q-supervised opponent-model DQN",
        "layer": "victim_q_supervision",
        "predicate": _and(_contains("results/pc_new_architectures_100k/victim_aware_dqn/seed_"), _eq("architecture", "dqn_victim_aware"), _eq("steps", 100_000)),
        "train_horizon_steps": "100,000",
        "eval_metric_prefix": "final_eval_continuation_adaptive",
        "initial_condition": "Own-run final checkpoint",
        "manuscript_use": "Victim-Q-supervised diagnostic",
        "information_set": "Public runtime features plus Victim-Q-derived auxiliary labels",
    },
    {
        "study": "100k privileged architecture diagnostics",
        "cell": "Q-row reconstruction DQN",
        "layer": "victim_q_supervision",
        "predicate": _and(_contains("results/pc_new_architectures_100k/qdecoder_normalized_q/seed_"), _eq("architecture", "dqn_shared_jepa_qdecoder"), _eq("steps", 100_000)),
        "train_horizon_steps": "100,000",
        "eval_metric_prefix": "final_eval_continuation_adaptive",
        "initial_condition": "Own-run final checkpoint",
        "manuscript_use": "Victim-Q-supervised representation diagnostic",
        "information_set": "Public observations plus Victim-Q auxiliary target",
    },
    {
        "study": "150k privileged architecture diagnostics",
        "cell": "Model-based Q-update rollout planner, H=5",
        "layer": "victim_model_access",
        "predicate": _and(_contains("results/long_matrix_100k_plus/block4_rollout_lola_150k/horizon_5/seed_"), _eq("architecture", "tabular_rollout_lola"), _eq("steps", 150_000)),
        "train_horizon_steps": "150,000",
        "eval_metric_prefix": "final_eval",
        "initial_condition": "Own-run final checkpoint",
        "manuscript_use": "Victim-model-access diagnostic; default/fresh eval metrics, not continuation-adaptive",
        "information_set": "Public state plus Victim Q-update rollout model",
    },
]


BEHAVIORAL_SIGNATURES = [
    {
        "probe": "DQN controls",
        "signature": "lower-price basin; destructive undercutting",
        "manuscript_role": "Generic learned control",
    },
    {
        "probe": "Tabular full-information regret matching",
        "signature": "stable low-price undercutting",
        "manuscript_role": "Counterfactual-value control",
    },
    {
        "probe": "Latent-prediction DQN (JEPA-style)",
        "signature": "representation gain without Q-vs-Q recovery",
        "manuscript_role": "Main learned architecture probe",
    },
    {
        "probe": "Q-supervised opponent-model DQN",
        "signature": "opponent-awareness/price-umbrella behavior in some runs; no Q-vs-Q recovery",
        "manuscript_role": "Main learned architecture probe with privileged-supervision caveat",
    },
    {
        "probe": "Q-row reconstruction DQN",
        "signature": "Q-structure auxiliary target without Q-vs-Q recovery",
        "manuscript_role": "Privileged-supervision diagnostic",
    },
    {
        "probe": "Model-based Q-update rollout planner",
        "signature": "model-based rollout below Q-vs-Q reference under available default evaluation",
        "manuscript_role": "Privileged-rollout diagnostic",
    },
]


def registry() -> pd.DataFrame:
    path = ROOT / "EXPERIMENT_REGISTRY.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment registry: {path}")
    df = pd.read_csv(path)
    return df[df["status"].eq("completed")].copy()


def cell_frame(reg: pd.DataFrame, cell: dict) -> pd.DataFrame:
    df = reg.loc[cell["predicate"](reg)].copy()
    if df.empty:
        return df
    df = df.sort_values(["seed", "result_path"]).drop_duplicates(subset=["seed", "result_path"])
    return df


def metric_columns(prefix: str) -> dict[str, str]:
    return {
        "oracle_profit": f"{prefix}_avg_profit_oracle",
        "victim_profit": f"{prefix}_avg_profit_victim",
        "market_price": f"{prefix}_market_price_mean",
        "oracle_price": f"{prefix}_avg_price_oracle",
        "victim_price": f"{prefix}_avg_price_victim",
        "victim_epsilon": f"{prefix}_victim_avg_epsilon",
    }


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 7:
        return math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(BOOTSTRAP_DRAWS)])
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def bootstrap_difference_ci(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    if len(left) < 7 or len(right) < 7:
        return math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    left_draws = left[rng.integers(0, len(left), size=(BOOTSTRAP_DRAWS, len(left)))].mean(axis=1)
    right_draws = right[rng.integers(0, len(right), size=(BOOTSTRAP_DRAWS, len(right)))].mean(axis=1)
    lo, hi = np.quantile(left_draws - right_draws, [0.025, 0.975])
    return float(lo), float(hi)


def exact_permutation_p_value(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return math.nan
    observed = abs(float(left.mean() - right.mean()))
    combined = np.concatenate([left, right])
    n_left = len(left)
    n_right = len(right)
    total_sum = float(combined.sum())
    extreme = 0
    total = 0
    for selected in itertools.combinations(range(len(combined)), n_left):
        selected_sum = float(combined[list(selected)].sum())
        difference = abs(selected_sum / n_left - (total_sum - selected_sum) / n_right)
        extreme += difference >= observed - 1e-15
        total += 1
    return extreme / total


def holm_adjust(p_values: pd.Series) -> pd.Series:
    adjusted = pd.Series(math.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    running = 0.0
    m = len(valid)
    for rank, (index, value) in enumerate(valid.items()):
        running = max(running, min(1.0, (m - rank) * float(value)))
        adjusted.loc[index] = running
    return adjusted


def mature_checkpoint_dir(seed: int) -> Path:
    if seed == 0:
        return ROOT / "results" / "mature_q_vs_q_checkpoint_10m"
    return ROOT / "results" / f"mature_q_vs_q_checkpoint_10m_seed_{seed}"


def mature_reference_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed in range(10):
        summary = json.loads((mature_checkpoint_dir(seed) / "summary.json").read_text(encoding="utf-8"))
        final = summary["final_eval"]
        rows.append(
            {
                "seed": seed,
                "symmetric_profit": float(final["symmetric_profit"]),
                "market_price": float(final["market_price"]),
                "price_agent_0": float(final["price_agent_0"]),
                "price_agent_1": float(final["price_agent_1"]),
                "profit_agent_0": float(final["profit_agent_0"]),
                "profit_agent_1": float(final["profit_agent_1"]),
                "cycle_length": int(final["cycle_length"]),
            }
        )
    by_seed = pd.DataFrame(rows)
    reference_prices = by_seed[["price_agent_0", "price_agent_1"]].rename(
        columns={"price_agent_0": "oracle_price", "price_agent_1": "victim_price"}
    )
    reference_welfare = _static_welfare(reference_prices)
    by_seed["consumer_surplus"] = reference_welfare["consumer_surplus"]
    by_seed["joint_profit"] = by_seed["profit_agent_0"] + by_seed["profit_agent_1"]
    by_seed["total_welfare"] = by_seed["consumer_surplus"] + by_seed["joint_profit"]
    profit_ci = bootstrap_ci(by_seed["symmetric_profit"].to_numpy())
    price_ci = bootstrap_ci(by_seed["market_price"].to_numpy())
    aggregate = pd.DataFrame(
        [
            {
                "N_seed": len(by_seed),
                "symmetric_profit_mean": by_seed["symmetric_profit"].mean(),
                "symmetric_profit_sample_sd": by_seed["symmetric_profit"].std(ddof=1),
                "symmetric_profit_bootstrap_ci_low": profit_ci[0],
                "symmetric_profit_bootstrap_ci_high": profit_ci[1],
                "market_price_mean": by_seed["market_price"].mean(),
                "market_price_sample_sd": by_seed["market_price"].std(ddof=1),
                "market_price_bootstrap_ci_low": price_ci[0],
                "market_price_bootstrap_ci_high": price_ci[1],
                "consumer_surplus_mean": by_seed["consumer_surplus"].mean(),
                "joint_profit_mean": by_seed["joint_profit"].mean(),
                "total_welfare_mean": by_seed["total_welfare"].mean(),
                "statistical_unit": "independently trained mature checkpoint seed",
            }
        ]
    )
    return by_seed, aggregate


def stats_table(
    reg: pd.DataFrame,
    q_profit_values: np.ndarray,
    q_price_values: np.ndarray,
) -> pd.DataFrame:
    q_profit = float(q_profit_values.mean())
    rows = []
    for cell in CELLS:
        df = cell_frame(reg, cell)
        cols = metric_columns(cell["eval_metric_prefix"])
        profits = pd.to_numeric(df.get(cols["oracle_profit"], pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
        victims = pd.to_numeric(df.get(cols["victim_profit"], pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
        prices = pd.to_numeric(df.get(cols["market_price"], pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
        differences = profits - q_profit
        below = int(np.sum(differences < 0))
        ci_lo, ci_hi = bootstrap_difference_ci(profits, q_profit_values)
        price_differences = prices - float(q_price_values.mean())
        price_ci_lo, price_ci_hi = bootstrap_difference_ci(prices, q_price_values)
        rows.append(
            {
                "layer": cell["layer"],
                "study": cell["study"],
                "cell": cell["cell"],
                "N_seed": int(len(profits)),
                "seeds": ",".join(str(int(seed)) for seed in sorted(df["seed"].dropna().unique())),
                "eval_metric_prefix": cell["eval_metric_prefix"],
                "oracle_profit_mean": float(np.mean(profits)) if len(profits) else math.nan,
                "oracle_profit_sample_sd": float(np.std(profits, ddof=1)) if len(profits) > 1 else math.nan,
                "victim_profit_mean": float(np.mean(victims)) if len(victims) else math.nan,
                "market_price_mean": float(np.mean(prices)) if len(prices) else math.nan,
                "mean_difference_vs_q_vs_q": float(np.mean(differences)) if len(differences) else math.nan,
                "profit_difference_vs_q_vs_q_mean": float(np.mean(differences)) if len(differences) else math.nan,
                "seeds_below_reference": below,
                "all_seeds_below_reference_mean": bool(len(differences) and below == len(differences)),
                "exact_permutation_p_two_sided": exact_permutation_p_value(profits, q_profit_values),
                "profit_exact_permutation_p_two_sided": exact_permutation_p_value(profits, q_profit_values),
                "profit_holm_p_all_tested": math.nan,
                "bootstrap_mean_difference_ci_low": ci_lo,
                "bootstrap_mean_difference_ci_high": ci_hi,
                "profit_bootstrap_difference_ci_low": ci_lo,
                "profit_bootstrap_difference_ci_high": ci_hi,
                "market_price_difference_vs_q_vs_q_mean": (
                    float(np.mean(price_differences)) if len(price_differences) else math.nan
                ),
                "market_price_exact_permutation_p_two_sided": exact_permutation_p_value(
                    prices,
                    q_price_values,
                ),
                "market_price_holm_p_all_tested": math.nan,
                "market_price_bootstrap_difference_ci_low": price_ci_lo,
                "market_price_bootstrap_difference_ci_high": price_ci_hi,
                "inference_note": (
                    "No p-value; N<2"
                    if len(profits) < 2
                    else (
                        "Permutation tests target identical learner/reference distributions "
                        "under exchangeability; learner run and mature checkpoint are the units"
                    )
                ),
                "manuscript_use": cell["manuscript_use"],
                "information_set": cell["information_set"],
            }
        )
    result = pd.DataFrame(rows)
    result["profit_holm_p_all_tested"] = holm_adjust(
        result["profit_exact_permutation_p_two_sided"]
    )
    result["market_price_holm_p_all_tested"] = holm_adjust(
        result["market_price_exact_permutation_p_two_sided"]
    )
    result["holm_p_main_claims"] = result["profit_holm_p_all_tested"]
    return result


def read_config(path: str) -> dict:
    config_path = ROOT / path / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def read_summary(path: str) -> dict:
    summary_path = ROOT / path / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def provenance_table(reg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell in CELLS:
        df = cell_frame(reg, cell)
        config = read_config(str(df.iloc[0]["result_path"])) if len(df) else {}
        cols = metric_columns(cell["eval_metric_prefix"])
        eps = pd.to_numeric(df.get(cols["victim_epsilon"], pd.Series(dtype=float)), errors="coerce").dropna()
        if eps.empty and len(df):
            eps_values = [
                read_summary(str(path)).get(cols["victim_epsilon"])
                for path in df["result_path"]
            ]
            eps = pd.to_numeric(pd.Series(eps_values), errors="coerce").dropna()
        rows.append(
            {
                "layer": cell["layer"],
                "study": cell["study"],
                "cell": cell["cell"],
                "source": "EXPERIMENT_REGISTRY.csv",
                "train_horizon_steps": cell["train_horizon_steps"],
                "seeds": ",".join(str(int(seed)) for seed in sorted(df["seed"].dropna().unique())),
                "N_seed": int(df["seed"].nunique()) if len(df) else 0,
                "initial_condition": cell["initial_condition"],
                "eval_metric_prefix": cell["eval_metric_prefix"],
                "eval_mode": "continuation_adaptive" if cell["eval_metric_prefix"].endswith("continuation_adaptive") else "default/fresh final_eval",
                "eval_victim_epsilon_mean": float(eps.mean()) if len(eps) else math.nan,
                "B_vectorized_envs": config.get("B", math.nan),
                "eval_steps": config.get("eval_steps", math.nan),
                "statistical_unit": "seed",
                "rollout_unit_note": "B vectorized envs are rollout batches, not independent replications",
                "benchmark_role": "Compared to fixed mature 10M Q-vs-Q reference",
                "manuscript_use": cell["manuscript_use"],
                "information_set": cell["information_set"],
            }
        )
    return pd.DataFrame(rows)


def _static_welfare(prices: pd.DataFrame) -> pd.DataFrame:
    config = CalvanoMarketConfig()
    qualities, costs = market_arrays(config)
    rows = []
    for row in prices.itertuples(index=False):
        price_pair = np.array([[float(row.oracle_price), float(row.victim_price)]], dtype=float)
        profits, shares, _ = profit_vector(
            price_pair,
            qualities,
            costs,
            config.outside_quality,
            config.mu,
            config.demand_scale,
        )
        inclusive_value = (
            np.exp(config.outside_quality / config.mu)
            + np.exp((qualities[0] - price_pair[0, 0]) / config.mu)
            + np.exp((qualities[1] - price_pair[0, 1]) / config.mu)
        )
        consumer_surplus = config.mu * np.log(inclusive_value)
        rows.append(
            {
                "oracle_profit_static": float(profits[0, 0]),
                "victim_profit_static": float(profits[0, 1]),
                "oracle_share": float(shares[0, 0]),
                "victim_share": float(shares[0, 1]),
                "consumer_surplus": float(consumer_surplus),
            }
        )
    return pd.DataFrame(rows)


def welfare_table(reg: pd.DataFrame, reference_summary: pd.DataFrame) -> pd.DataFrame:
    reference = reference_summary.iloc[0]
    q_consumer_surplus = float(reference["consumer_surplus_mean"])
    q_total_welfare = float(reference["total_welfare_mean"])
    q_price = float(reference["market_price_mean"])

    rows = []
    for cell in CELLS:
        df = cell_frame(reg, cell)
        cols = metric_columns(cell["eval_metric_prefix"])
        if not len(df):
            continue
        summary_rows = []
        for result_path in df["result_path"]:
            summary = read_summary(str(result_path))
            if not summary:
                continue
            summary_rows.append(
                {
                    cols["oracle_price"]: summary.get(cols["oracle_price"]),
                    cols["victim_price"]: summary.get(cols["victim_price"]),
                    cols["oracle_profit"]: summary.get(cols["oracle_profit"]),
                    cols["victim_profit"]: summary.get(cols["victim_profit"]),
                    cols["market_price"]: summary.get(cols["market_price"]),
                }
            )
        if not summary_rows:
            continue
        usable = pd.DataFrame(summary_rows).apply(pd.to_numeric, errors="coerce").dropna()
        if usable.empty:
            continue
        prices = usable[[cols["oracle_price"], cols["victim_price"]]].rename(
            columns={cols["oracle_price"]: "oracle_price", cols["victim_price"]: "victim_price"}
        )
        welfare = _static_welfare(prices)
        enriched = pd.concat([usable.reset_index(drop=True), welfare], axis=1)
        enriched["joint_profit_realized"] = enriched[cols["oracle_profit"]] + enriched[cols["victim_profit"]]
        enriched["total_welfare_realized_profit"] = enriched["consumer_surplus"] + enriched["joint_profit_realized"]
        rows.append(
            {
                "layer": cell["layer"],
                "cell": cell["cell"],
                "source": "EXPERIMENT_REGISTRY.csv",
                "N_seed": int(len(enriched)),
                "eval_metric_prefix": cell["eval_metric_prefix"],
                "source_type": "final_eval_aggregate_prices_not_full_trajectory",
                "consumer_surplus_gain_vs_q_vs_q_mean": float((enriched["consumer_surplus"] - q_consumer_surplus).mean()),
                "consumer_surplus_gain_vs_q_vs_q_sample_sd": float((enriched["consumer_surplus"] - q_consumer_surplus).std(ddof=1))
                if len(enriched) > 1
                else math.nan,
                "total_welfare_gain_vs_q_vs_q_mean": float((enriched["total_welfare_realized_profit"] - q_total_welfare).mean()),
                "total_welfare_gain_vs_q_vs_q_sample_sd": float((enriched["total_welfare_realized_profit"] - q_total_welfare).std(ddof=1))
                if len(enriched) > 1
                else math.nan,
                "market_price_minus_q_vs_q_mean": float((enriched[cols["market_price"]] - q_price).mean()),
                "manuscript_use": cell["manuscript_use"],
            }
        )
    return pd.DataFrame(rows)


def write_outputs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = registry()
    reference_by_seed, reference_summary = mature_reference_tables()
    q_profit_values = reference_by_seed["symmetric_profit"].to_numpy()
    q_price_values = reference_by_seed["market_price"].to_numpy()
    reference_by_seed.to_csv(out_dir / "q_vs_q_reference_by_seed.csv", index=False)
    reference_summary.to_csv(out_dir / "q_vs_q_reference_summary.csv", index=False)
    stats = stats_table(reg, q_profit_values, q_price_values)
    stats.to_csv(out_dir / "stats_table.csv", index=False)
    stats[stats["layer"].isin(["public_bandit_feedback", "known_payoff_model"])].to_csv(
        out_dir / "main_learned_stats.csv",
        index=False,
    )
    stats[stats["layer"].isin(["victim_q_supervision", "victim_model_access"])].to_csv(
        out_dir / "privileged_architecture_diagnostics.csv",
        index=False,
    )
    provenance_table(reg).to_csv(out_dir / "denominator_provenance.csv", index=False)
    welfare_table(reg, reference_summary).to_csv(out_dir / "architecture_welfare_current.csv", index=False)
    pd.DataFrame(BEHAVIORAL_SIGNATURES).to_csv(out_dir / "behavioral_signatures.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="paper_preprint/generated_tables",
        help="Directory for generated CSV tables.",
    )
    args = parser.parse_args()
    write_outputs(ROOT / args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
