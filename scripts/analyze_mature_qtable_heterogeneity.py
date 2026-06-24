"""Analyze on-path agreement and off-path heterogeneity in mature Q-tables."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "mature_qtable_heterogeneity"
SEEDS = range(10)


def checkpoint_dir(seed: int) -> Path:
    if seed == 0:
        return ROOT / "results" / "mature_q_vs_q_checkpoint_10m"
    return ROOT / "results" / f"mature_q_vs_q_checkpoint_10m_seed_{seed}"


def load_seed(seed: int) -> tuple[np.ndarray, dict, int, int]:
    directory = checkpoint_dir(seed)
    with np.load(directory / "mature_victim_state.npz") as state:
        q = np.asarray(state["Q"], dtype=np.float64).mean(axis=0)
        terminal_state = int(np.asarray(state["state_id"]).reshape(-1)[0])
    summary = json.loads((directory / "summary.json").read_text())
    greedy_action = int(np.argmax(q[terminal_state]))
    return q, summary, terminal_state, greedy_action


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = {seed: load_seed(seed) for seed in SEEDS}

    seed_rows = []
    for seed, (q, summary, terminal_state, greedy_action) in loaded.items():
        final = summary["final_eval"]
        seed_rows.append(
            {
                "seed": seed,
                "terminal_state": terminal_state,
                "terminal_greedy_action": greedy_action,
                "cycle_length": int(final["cycle_length"]),
                "symmetric_profit": float(final["symmetric_profit"]),
                "market_price": float(final["market_price"]),
                "terminal_selected_q": float(q[terminal_state, greedy_action]),
                "q_mean": float(q.mean()),
                "q_sd": float(q.std()),
            }
        )
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(OUT / "seed_summary.csv", index=False)

    pair_rows = []
    for left, right in combinations(SEEDS, 2):
        q_left, summary_left, state_left, action_left = loaded[left]
        q_right, summary_right, state_right, action_right = loaded[right]
        final_left = summary_left["final_eval"]
        final_right = summary_right["final_eval"]
        q_diff = q_left - q_right
        same_terminal_policy = state_left == state_right and action_left == action_right
        selected_abs_diff = np.nan
        row_alternative_rmse = np.nan
        if same_terminal_policy:
            selected_abs_diff = abs(q_left[state_left, action_left] - q_right[state_right, action_right])
            alternatives = np.arange(q_left.shape[1]) != action_left
            row_alternative_rmse = float(
                np.sqrt(np.mean((q_left[state_left, alternatives] - q_right[state_right, alternatives]) ** 2))
            )
        pair_rows.append(
            {
                "seed_left": left,
                "seed_right": right,
                "same_terminal_policy": same_terminal_policy,
                "profit_abs_diff": abs(float(final_left["symmetric_profit"]) - float(final_right["symmetric_profit"])),
                "price_abs_diff": abs(float(final_left["market_price"]) - float(final_right["market_price"])),
                "q_table_rmse": float(np.sqrt(np.mean(q_diff**2))),
                "greedy_policy_disagreement_rate": float(
                    np.mean(np.argmax(q_left, axis=1) != np.argmax(q_right, axis=1))
                ),
                "terminal_selected_q_abs_diff": selected_abs_diff,
                "terminal_row_alternative_rmse": row_alternative_rmse,
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(OUT / "pairwise_distances.csv", index=False)

    # Seeds 0 and 8 have the same terminal state, greedy action, price, and profit.
    left, right = 0, 8
    q_left, _, state, action = loaded[left]
    q_right, _, _, _ = loaded[right]
    abs_diff = np.abs(q_left - q_right)
    selected_diff = float(abs_diff[state, action])
    row_alt = np.delete(abs_diff[state], action)
    other_states = np.delete(abs_diff, state, axis=0).ravel()
    matched = pd.DataFrame(
        [
            {"region": "terminal selected cell", "count": 1, "mean_abs_diff": selected_diff, "rmse": selected_diff},
            {
                "region": "terminal-row alternative actions",
                "count": len(row_alt),
                "mean_abs_diff": float(row_alt.mean()),
                "rmse": float(np.sqrt(np.mean(row_alt**2))),
            },
            {
                "region": "all other states/actions",
                "count": len(other_states),
                "mean_abs_diff": float(other_states.mean()),
                "rmse": float(np.sqrt(np.mean(other_states**2))),
            },
        ]
    )
    matched.to_csv(OUT / "seed_0_vs_8_on_off_path.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].scatter(seed_df["market_price"], seed_df["symmetric_profit"], color="#2563eb", s=55)
    for row in seed_df.itertuples():
        axes[0].annotate(f"s{row.seed}", (row.market_price, row.symmetric_profit), xytext=(4, 3), textcoords="offset points")
    axes[0].set_xlabel("Final market price")
    axes[0].set_ylabel("Final symmetric profit")
    axes[0].set_title("Mature outcomes across seeds")

    image = axes[1].imshow(abs_diff.T, aspect="auto", cmap="magma")
    axes[1].scatter([state], [action], marker="s", facecolors="none", edgecolors="#22c55e", s=90, linewidths=1.5)
    axes[1].set_xlabel("State")
    axes[1].set_ylabel("Victim action")
    axes[1].set_title("Absolute Q difference: seeds 0 vs 8")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    regions = ["selected", "row alternatives", "other states"]
    values = [
        selected_diff,
        float(row_alt.mean()),
        float(other_states.mean()),
    ]
    axes[2].bar(regions, values, color=["#22c55e", "#f59e0b", "#7c3aed"])
    axes[2].set_ylabel("Mean absolute Q difference")
    axes[2].set_title("Exact outcome match, different off-path values")
    axes[2].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(OUT / "mature_qtable_heterogeneity.png", dpi=300)
    plt.close(fig)

    report = f"""# Mature Q-Table Heterogeneity Diagnostic

This diagnostic uses the ten independently trained 10M Q-vs-Q checkpoints. The
seed is the statistical unit; the 64 vectorized replicas inside a checkpoint
are averaged and are not treated as independent observations.

## Exact Matched-Outcome Case

Seeds 0 and 8 end at the same terminal state ({state}), choose the same greedy
Victim action ({action}), and have identical final market price and symmetric
profit. Their selected on-path Q-value is also identical at saved precision.
Nevertheless, their alternative actions in the terminal row and the rest of
the Q-table differ:

- selected-cell absolute difference: {selected_diff:.8f};
- terminal-row alternative-action RMSE: {np.sqrt(np.mean(row_alt**2)):.6f};
- all-other-state/action RMSE: {np.sqrt(np.mean(other_states**2)):.6f};
- all-state greedy-policy disagreement rate:
  {np.mean(np.argmax(q_left, axis=1) != np.argmax(q_right, axis=1)):.3f}.

This is a descriptive identification example, not a population theorem:
behavioral agreement on the recurrent path does not identify a unique global
Q-table. The unreached portion retains seed-specific learning history.

## Outputs

- `seed_summary.csv`
- `pairwise_distances.csv`
- `seed_0_vs_8_on_off_path.csv`
- `mature_qtable_heterogeneity.png`
"""
    (OUT / "REPORT.md").write_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
