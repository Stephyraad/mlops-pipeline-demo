# Monitoring: Data Drift

## What's being monitored

Every call to `train_model()` ([src/train_model.py](src/train_model.py)) generates a data drift report via `build_drift_report()` ([src/monitor_drift.py](src/monitor_drift.py)), using Evidently's `DataDriftPreset`. The comparison is:

- **Reference data**: `features_train` — the training split
- **Current data**: `features_val` — the validation split (`test_size=0.2`, `random_state=1234`)

Both splits come from the same snapshot of `employee_attrition_dataset.csv` at training time — this is **not** a comparison against new incoming/production data (the pipeline has no live data source yet). It's effectively a **split-representativeness check**: are train and validation drawn from the same distribution, or did the random split introduce skew that could bias evaluation?

Reports are written to `reports/drift_report_<timestamp>.html` on every run. That directory is gitignored, so nothing is retained between runs or machines — this file is the durable record of what those reports have shown.

## Methodology

Evidently picks a per-column test based on type:

| Feature type | Test | Drift threshold |
|---|---|---|
| Numerical | Kolmogorov–Smirnov (K-S) p-value | p < 0.05 |
| Categorical (>2 categories) | Chi-square p-value | p < 0.05 |
| Binary/categorical (2 categories) | Z-test p-value | p < 0.05 |

A column is flagged as drifted when its test p-value falls below 0.05. The dataset is flagged as having **overall drift** only when the share of drifted columns reaches `drift_share=0.5` (i.e. at least half of all features).

## Latest run results

Out of 24 features:

- **Drifted columns: 3 / 24 (12.5%)** — well below the 50% dataset-level drift threshold, so no overall drift alarm.

| Column | Test | p-value |
|---|---|---|
| Department | chi-square | 0.0075 |
| Number_of_Companies_Worked | chi-square | 0.0432 |
| Hourly_Rate | K-S | 0.0473 |

Everything else (Age, Monthly_Income, Years_at_Company, Job_Satisfaction, Gender, Overtime, Job_Level, Work_Life_Balance, etc.) showed p-values well above 0.05 (many > 0.5), i.e. train and validation look like the same distribution.

## Interpretation

- The two borderline categorical results (`Department`, `Number_of_Companies_Worked`) and the marginal numerical one (`Hourly_Rate`, p = 0.047 — just under the 0.05 cutoff) are consistent with what you'd expect from a single random 80/20 split of ~1,000 rows: some features will cross a p < 0.05 threshold by chance alone, especially categorical ones with several low-count categories (chi-square is sensitive to sparse bins at this sample size).
- With `drift_share=0.5` unmet, the pipeline correctly treats this run as "no meaningful drift" — the split is usable for training/evaluation.
- `Overtime` (p = 0.067) is the closest non-drifted feature to the cutoff and worth watching if it flips in future runs.

## Caveats / limitations

- **Not production monitoring.** Because reference and current are both slices of one static dataset, this only catches *split* skew, not real-world *data* drift (e.g. a new hiring cohort, a policy change affecting `Overtime`, a pay-scale update affecting `Monthly_Income`/`Hourly_Rate`). There's currently no mechanism to compare against a fixed reference (e.g. the original training set) against newly-arriving batches.
- **No retention or diffing.** Every run produces a new timestamped HTML file locally; nothing aggregates results across runs or alerts when `drift_share` trends upward.
- **No CI gate.** `train_model()` generates the report but doesn't act on it — a high-drift run does not fail the pipeline or block a model from being logged to MLflow.

## Suggested next steps

1. Once real usage data is available, split monitoring into two modes: (a) train/val split check (as today, a sanity check), and (b) reference (training set) vs. live incoming data (actual drift detection).
2. Persist a rollup (e.g. `metrics/drift_summary.json`, appended per run) so drift share can be tracked over time instead of only living in disposable HTML files.
3. Add a CI/pipeline check that fails or warns when `drift_share` crosses a set threshold, rather than only generating a report nobody looks at automatically.
