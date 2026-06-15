# Prediction_b Two-Metric Analysis

This folder contains a standalone implementation for reviewer-focused quantitative validation on issue34 Prediction_b using two non-bond metrics:

1. Hausdorff distance
2. Matching distance

## Script

- `run_prediction_b_two_metrics.py`

## Model Mapping

- `Ref_Pure_C9_issue34` -> `F_U`
- `Ref_issue34` -> `F_Vbar`
- `PPAFM2Exp_CoAll_L20_L1_Elatest_Only_C7_issue34` -> `F_Vtilde`
- `PPAFM2Exp_CoAll_L20_L1_Elatest_C1_issue34` -> `F_Vdagger`

## Run

From this directory:

```bash
conda run -n ml-spm python run_prediction_b_two_metrics.py
```

## Outputs

Generated under `outputs/`:

- `per_sample_metrics.csv`
- `summary_metrics.csv`
- `paired_tests.csv`
- `seq_stats_comparison.csv`
- `plot_A_distributions.png`
- `plot_B_paired_differences.png`
- `plot_C_ecdf.png`

## Recommended Use in Manuscript Workflow

- SI figure set: use Plot A and Plot C as the main quantitative visualization pair.
- Reviewer-response figure set: use Plot B as paired, sample-matched supporting evidence.
- Rationale: A + C summarize distribution-level behavior clearly, while B provides within-sample consistency checks when reviewer requests paired validation.

## Notes

- The script uses shared sample IDs available in each `Prediction_b/predictions` directory.
- In the current repository snapshot, this corresponds to 48 paired xyz samples per model.
- This is a subset of the full validation total reported by `stats/seq_stats.csv` (360 samples), so `seq_stats_comparison.csv` is included for transparency.
- Distances are lower-is-better.
- In Plot B (`First - Second`), negative values indicate the first model has lower distance.
