# Ablation Studies

This folder stores outputs from ablation experiments (e.g., threshold sweeps, loss variations, neutral handling). Summary results are reported in `docs/results/model_performance.md`; use this folder only if you regenerate ablations locally.

## Where to find summaries
- Primary metrics and comparisons: `docs/results/model_performance.md`
- Design context: `docs/reference/design_decisions.md`
- Visualization scripts: `src/analysis/*`, `src/visualization/*`

## If you rerun ablations
1. Write a short summary table here (baseline vs variant, ΔAUC/ΔF1).
2. Drop CSVs/plots into this directory (gitignored if large).
3. Link any W&B artifacts (project: `Cooper-Union/GoEmotions_Classification`) with the run ID or alias you used.

## Artifact retrieval (W&B)
- Use the W&B UI or `wandb artifact get` with the project above.
- Tag runs with `ablation` to keep them discoverable.

If this folder is empty, no local ablation outputs are currently stored. Summaries remain in the docs/results page.
