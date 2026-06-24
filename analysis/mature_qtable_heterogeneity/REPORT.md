# Mature Q-Table Heterogeneity Diagnostic

This diagnostic uses the ten independently trained 10M Q-vs-Q checkpoints. The
seed is the statistical unit; the 64 vectorized replicas inside a checkpoint
are averaged and are not treated as independent observations.

## Exact Matched-Outcome Case

Seeds 0 and 8 end at the same terminal state (130), choose the same greedy
Victim action (10), and have identical final market price and symmetric
profit. Their selected on-path Q-value is also identical at saved precision.
Nevertheless, their alternative actions in the terminal row and the rest of
the Q-table differ:

- selected-cell absolute difference: 0.00000000;
- terminal-row alternative-action RMSE: 0.092533;
- all-other-state/action RMSE: 0.101952;
- all-state greedy-policy disagreement rate:
  0.933.

This is a descriptive identification example, not a population theorem:
behavioral agreement on the recurrent path does not identify a unique global
Q-table. The unreached portion retains seed-specific learning history.

## Outputs

- `seed_summary.csv`
- `pairwise_distances.csv`
- `seed_0_vs_8_on_off_path.csv`
- `mature_qtable_heterogeneity.png`
