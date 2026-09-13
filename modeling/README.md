# Baseline modeling (acoustic + MFCC + temporal, Channel 0 only)

The first non-NLP baseline: human-vs-synthetic classification for
Channel 0, using only the already-computed acoustic (`../preprocessing/`),
MFCC, and temporal (`../temporal/`) recording-level features. See
`../rules.md` for project rules. Simple models, no tuning, no deep
learning — the goal is to understand the data, not ship a detector.

## Read this before trusting the numbers

**The baseline models score at or near a perfect ROC-AUC (1.0) on both
train and val.** This was investigated thoroughly — see
**`outputs/reports/modeling_evaluation_report.md`** (produced by
`modeling/diagnostics.py`) for the full write-up: feature-family
ablation, controlled feature elimination, univariate separability,
recording-condition analysis, original-vs-denoised comparison, and a
Channel-1 (confirmed AI agent) cross-check. Short version: the
separation is driven mainly by spectral-shape/zero-crossing-rate
features (not pitch, not simple loudness alone), survives denoising,
and shows up independently in Channel 1 too — consistent with a real
but *possibly pipeline-specific* acoustic signature rather than a
confirmed, generalizable detector. Treat every baseline number here as
**a description of this dataset**, not a working detector, until the
recommendations at the end of that report are addressed.

### Diagnostics module

- `cv_utils.py` — shared, leakage-safe evaluation helpers (stratified
  CV with scaling fit inside each fold via `sklearn.Pipeline`, val
  scoring with the scaler fit on train only).
- `diagnostics.py` — every experiment in `modeling_evaluation_report.md`
  (feature-family ablation, controlled elimination, univariate
  separability + plots, recording-condition analysis, original-vs-
  denoised, Channel-1 diagnostic). Run with `python -m modeling.diagnostics`
  after `modeling.prepare_dataset`. Also runs
  `preprocessing/channel1_diagnostic_features.py` output as an input
  (run that once first — it's a fast, non-pitch feature extraction for
  Channel 1 only, ~15s for all 353 recordings).

## Running

```bash
source .venv/bin/activate
python -m preprocessing.mfcc_pipeline   # adds MFCC summaries (~13-14 recordings/sec on CPU)
python -m modeling.prepare_dataset      # joins acoustic + MFCC + temporal -> modeling_dataset*.csv
python -m modeling.baseline             # trains LR / RF / SVM
python -m modeling.evaluate             # metrics, confusion matrices, feature separability, plots
```

## Files

- `prepare_dataset.py` — joins the three feature families on `anon_id`
  (Channel 0 only; `manifest.csv`'s label via the acoustic table is the
  source of truth). Validates the join doesn't duplicate rows
  (`validate="one_to_one"` + row-count assertions), checks that
  identity columns agree where both tables have them, and produces the
  missing-value / leakage reports below. Also writes
  `modeling_feature_families.csv` (feature -> acoustic/mfcc/temporal),
  used by `evaluate.py`'s family-level analysis.
- `baseline.py` — Logistic Regression (scaled), Random Forest, and SVM
  (RBF, scaled). Trains on `split == "train"`, predicts on both splits.
  `label`: 1 = synthetic, 0 = human.
- `evaluate.py` — accuracy/precision/recall/F1/ROC-AUC + confusion
  matrices (val is the headline split, train shown only to check
  over/under-fitting); a univariate feature-separability table (which
  single features separate the classes best, computed on train only);
  permutation importance per model (with a note on why it comes back
  near-zero when performance is already at the ceiling); and plots.

## Key design decisions

- **Only the 258 recordings with valid turn metadata** are used for the
  baseline (temporal features require turns) — `modeling_dataset.csv`
  keeps all 353 with real NaNs for the temporal columns;
  `modeling_dataset_complete_cases.csv` (258 rows) is what the models
  actually train on. See the missing-value section of
  `outputs/reports/modeling_data_quality_report.md` for exactly which
  columns are affected and why imputation was NOT used for that block
  (an entire feature family being NaN isn't something a per-column
  median can honestly stand in for).
- **`duration_s` is excluded from the feature set** (not from the saved
  CSVs) because the temporal-stage confounding checks already flagged
  recording duration as a possible confound between human and synthetic
  recordings — including it risks the model keying on file length
  rather than speech content.
- **Train/val split comes from `manifest.csv`** (via the acoustic
  table's `split` column), not re-randomized. The leakage check
  verifies `train anon_ids ∩ val anon_ids == ∅` and that every row is
  one whole recording (never a frame or a turn), so no conversation can
  contribute to both sides of the split.
- **Channel 1 (confirmed AI agent) is excluded from training rows** in
  this first classifier — it remains a reference/control group for the
  exploratory NLP-stage comparisons (`../nlp/`), not mixed into the
  Channel-0 human-vs-synthetic target.
- MFCCs were added as a new, additive function in
  `../preprocessing/features.py` (`extract_mfcc`/`mfcc_summary`) and run
  via a separate script (`../preprocessing/mfcc_pipeline.py`) rather
  than folded into the existing acoustic pipeline — this avoids
  re-running the already-completed acoustic batch (~25 minutes) just to
  add 26 summary columns.

## Outputs

- `outputs/features/mfcc_recording_level.csv` — MFCC 1-13 mean/std, per (anon_id, version)
- `outputs/features/modeling_dataset.csv` — all 353 recordings, joined, real NaNs preserved
- `outputs/features/modeling_dataset_complete_cases.csv` — the 258-row baseline table
- `outputs/features/modeling_feature_families.csv` — feature -> family lookup
- `outputs/features/modeling_predictions.csv` — per-recording predictions, all 3 models, both splits
- `outputs/features/modeling_feature_importance.csv`, `modeling_univariate_separability.csv`
- `outputs/reports/mfcc_processing_report.md`, `modeling_data_quality_report.md`, `modeling_baseline_report.md`
- `outputs/plots/modeling/` — ROC curves, confusion matrices, feature-family importance,
  and boxplots of the top separating features by label
