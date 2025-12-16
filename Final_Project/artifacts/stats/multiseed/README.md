# Multi-Seed Test Metrics

This directory contains test metrics from 3 DistilBERT training runs with different random seeds for robustness analysis.

## Files

- `seed13_metrics.json` - Metrics from seed 13 run (run ID: hcx0hper)
- `seed23_metrics.json` - Metrics from seed 23 run (run ID: xglzn8sy)
- `seed0_metrics.json` - Metrics from seed 0 run

## Metrics Summary

| Seed | AUC Macro | AUC Micro | F1 Macro | F1 Micro |
|------|-----------|-----------|----------|----------|
| 13   | 0.74393   | 0.88072   | 0.08799  | 0.34993  |
| 23   | 0.73783   | 0.87755   | 0.08510  | 0.34360  |
| 0    | 0.73136   | 0.87427   | 0.08364  | 0.33768  |

**Mean ± Std:**
- AUC Macro: 0.73771 ± 0.00629
- AUC Micro: 0.87751 ± 0.00323
- F1 Macro: 0.08558 ± 0.00221
- F1 Micro: 0.34374 ± 0.00613

## Purpose

These metrics are used for:
1. Computing mean and standard deviation across seeds
2. Statistical significance testing vs RoBERTa-Large baseline
3. Quantifying model variance and reliability

## Analysis Outputs

1. **Aggregated metrics**: `artifacts/stats/multiseed_summary.csv`
   - Mean and standard deviation for all metrics across 3 seeds
   - Generated with: `python -m src.analysis.aggregate_seeds --metrics-dir artifacts/stats/multiseed/ --model-pattern distilbert --output artifacts/stats/multiseed_summary.csv`

## Next Steps

1. Significance tests: `python -m src.analysis.significance_test`
2. Visualizations: Generate confidence interval plots

## W&B Run Links

- Seed 13: https://wandb.ai/Cooper-Union/GoEmotions_Classification/runs/hcx0hper
- Seed 23: https://wandb.ai/Cooper-Union/GoEmotions_Classification/runs/xglzn8sy
- Seed 0: https://wandb.ai/Cooper-Union/GoEmotions_Classification (check recent runs)
