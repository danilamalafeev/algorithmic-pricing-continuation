# Frozen-Opponent Evaluation and Live-Update Continuation in Algorithmic Pricing

Companion code and data for the working paper by **Danila Malafeev** (National
Research University Higher School of Economics).

The paper studies a Calvano-style Bertrand-logit duopoly and asks two questions:

- **RQ1.** How does a DQN entrant's post-training performance change when a
  mature tabular Q-learning incumbent is *frozen* versus *permitted to resume
  Q-updates*? This revisits heterogeneous-agent experiments (e.g., Deng,
  Schiffer, and Bichler, 2025) that train a new DRL agent against a pretrained,
  frozen TQL incumbent.
- **RQ2.** Is the high-price convention produced by mature Q-vs-Q self-play
  readily recovered by richer representation learning, counterfactual payoff
  feedback, or escalating access to incumbent information?

## Repository contents

```
paper_preprint/      manuscript source (main.tex), compiled PDF, and generated_tables/
paper_figures/       rendered figures included in the paper
analysis/            figure sources and the mature-Q-table heterogeneity diagnostics
results/             slim seed-level result tree (per-seed eval metrics + 10 mature Q-tables)
EXPERIMENT_REGISTRY.csv   seed-level experiment index (drives table generation)
calvano_market.py    Bertrand-logit environment (pure Python, NumPy)
calvano_qlearning.py tabular Q-learning incumbent
neural/              DQN and architecture-probe learners
src/                 optional C++ acceleration (market.cpp, bindings.cpp)
scripts/             reproduction and figure scripts
tests/               unit tests
```

The `results/` tree here is a **curated subset** (~5 MB) containing only the
seed-level metrics and the ten mature Q-vs-Q checkpoint tables needed to
reproduce the paper's tables and figures. The full raw-output archive (trajectories,
checkpoints, logs) is retained in the authors' private archive and is not
required for any reported number.

## Reproduce

Requirements: Python 3 with `numpy` and `pandas`; [tectonic](https://tectonic-typesetting.github.io/)
(or any LaTeX engine) to build the PDF.

```bash
# 1. Regenerate every table and the seed-level reference from bundled data
python scripts/reproduce_tables.py

# 2. Build the paper PDF
cd paper_preprint && tectonic main.tex
```

Sanity check: `reproduce_tables.py` writes `generated_tables/q_vs_q_reference_summary.csv`
with mature Q-vs-Q mean own profit **0.320496** and mean market price **1.784199**,
matching the constants in `main.tex`.

## Licensing

- **Code** (environment, learners, scripts, tests): MIT — see `LICENSE`.
- **Paper** (`paper_preprint/main.tex`, figures): Creative Commons Attribution
  4.0 International (CC-BY-4.0).

## Citation

```bibtex
@misc{malafeev2026pricing,
  title        = {Frozen-Opponent Evaluation and Live-Update Continuation in Algorithmic Pricing},
  author       = {Danila Malafeev},
  year         = {2026},
  note         = {Working paper, HSE University},
}
```
