"""
Builds the combined recording-level modeling table: acoustic + MFCC +
temporal features, Channel 0 only, one row per recording. Reuses the
existing acoustic/MFCC/temporal outputs — nothing is recomputed here.

manifest.csv (via the acoustic recording_level.csv, which already
carries it) remains the source of truth for the human/synthetic label.
Channel 1 (confirmed AI agent) is NOT included as a training row here —
see rules.md / the task instructions: it's a reference group for
exploratory comparison, not part of the first binary classifier.

Run with: python -m modeling.prepare_dataset
"""
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.pipeline import FEATURES_DIR as ACOUSTIC_FEATURES_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = REPO_ROOT / "outputs" / "features"
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"

ACOUSTIC_CSV = ACOUSTIC_FEATURES_DIR / "recording_level.csv"
MFCC_CSV = FEATURES_DIR / "mfcc_recording_level.csv"
TEMPORAL_CSV = FEATURES_DIR / "temporal_recording_level.csv"

MODELING_DATASET_CSV = FEATURES_DIR / "modeling_dataset.csv"
MODELING_DATASET_COMPLETE_CSV = FEATURES_DIR / "modeling_dataset_complete_cases.csv"
FEATURE_FAMILIES_CSV = FEATURES_DIR / "modeling_feature_families.csv"
DATA_QUALITY_REPORT_MD = REPORTS_DIR / "modeling_data_quality_report.md"

# non-feature identifier/metadata columns, plus duration_s: excluded from
# the modeling feature set (not from the saved CSV) since recording
# duration is a known potential confound (see temporal/compare.py's
# confounding checks) rather than a speech characteristic itself.
NON_FEATURE_COLS = ["anon_id", "label", "split", "has_turn_data", "duration_s"]

# columns from the acoustic recording_level.csv that identify the row or
# are constant/redundant for a Channel-0-only table (channel is always 0,
# sample_rate is always 8000, n_samples is redundant with duration_s) —
# dropped so the feature space doesn't carry zero-information columns.
ACOUSTIC_DROP_COLS = ["channel", "version", "sample_rate", "n_samples"]
# these come from BOTH the acoustic and temporal tables; keep the
# acoustic copy (present for all 353) and drop the temporal duplicate
# after checking the two agree wherever both exist.
TEMPORAL_DUPLICATE_COLS = ["label", "split", "duration_s", "has_turn_data"]


def load_acoustic() -> pd.DataFrame:
    df = pd.read_csv(ACOUSTIC_CSV)
    df = df[df["version"] == "original"].drop(columns=ACOUSTIC_DROP_COLS)
    return df


def load_mfcc() -> pd.DataFrame:
    df = pd.read_csv(MFCC_CSV)
    df = df[df["version"] == "original"].drop(columns=["version"])
    return df


def load_temporal() -> pd.DataFrame:
    return pd.read_csv(TEMPORAL_CSV)


def build_combined_table() -> tuple:
    acoustic = load_acoustic()
    mfcc = load_mfcc()
    temporal = load_temporal()

    n_acoustic, n_mfcc, n_temporal = len(acoustic), len(mfcc), len(temporal)
    assert acoustic["anon_id"].is_unique, "duplicate anon_id in acoustic table"
    assert mfcc["anon_id"].is_unique, "duplicate anon_id in MFCC table"
    assert temporal["anon_id"].is_unique, "duplicate anon_id in temporal table"

    combined = acoustic.merge(mfcc, on="anon_id", how="left", validate="one_to_one")
    assert len(combined) == n_acoustic, "acoustic-MFCC merge changed row count — possible duplicate join key"

    # sanity check: where both tables have a value for these identity
    # columns, they must agree (catches a mismatched/misaligned join,
    # not just duplicated rows)
    check_cols = [c for c in TEMPORAL_DUPLICATE_COLS if c in temporal.columns]
    merged_check = combined[["anon_id"] + [c for c in ["label", "split", "duration_s"] if c in combined.columns]].merge(
        temporal[["anon_id"] + check_cols], on="anon_id", how="inner", suffixes=("", "_temporal")
    )
    mismatches = {}
    for c in ("label", "split", "duration_s"):
        if c in merged_check.columns and f"{c}_temporal" in merged_check.columns:
            if c == "duration_s":
                bad = (merged_check[c] - merged_check[f"{c}_temporal"]).abs() > 0.5
            else:
                bad = merged_check[c] != merged_check[f"{c}_temporal"]
            if bad.any():
                mismatches[c] = int(bad.sum())

    # acoustic's has_turn_data is already correct and present for all 353
    # rows; drop temporal's copy too so the merge doesn't produce
    # has_turn_data_x/has_turn_data_y duplicate columns
    temporal_features_only = temporal.drop(columns=check_cols)
    combined = combined.merge(temporal_features_only, on="anon_id", how="left", validate="one_to_one")
    assert len(combined) == n_acoustic, "acoustic-temporal merge changed row count — possible duplicate join key"

    # feature-family lookup, for the baseline models' family-level
    # importance summary (modeling/evaluate.py) — built here, where each
    # source table's own column list is still on hand, rather than
    # re-derived later from column-name string patterns.
    mfcc_cols = [c for c in mfcc.columns if c != "anon_id"]
    temporal_cols = [c for c in temporal_features_only.columns if c != "anon_id"]
    acoustic_cols = [c for c in acoustic.columns if c not in ("anon_id", "label", "split", "has_turn_data", "duration_s")]
    families = (
        [{"feature": c, "family": "acoustic"} for c in acoustic_cols]
        + [{"feature": c, "family": "mfcc"} for c in mfcc_cols]
        + [{"feature": c, "family": "temporal"} for c in temporal_cols]
    )

    return combined, {
        "n_acoustic": n_acoustic, "n_mfcc": n_mfcc, "n_temporal": n_temporal,
        "n_combined": len(combined), "identity_mismatches": mismatches,
    }, pd.DataFrame(families)


def missing_value_report(df: pd.DataFrame) -> tuple:
    """
    Returns (report_markdown, complete_case_mask). Distinguishes the
    large, EXPECTED block of missingness (95 recordings with no turn
    metadata -> every temporal-derived column is NaN for them) from
    smaller, incidental NaNs elsewhere.
    """
    lines = ["## Missing-value inspection\n"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = int(np.isinf(df[numeric_cols].to_numpy()).sum())
    lines.append(f"Infinite values across all numeric columns: {n_inf}\n")

    no_turn_data = ~df["has_turn_data"].astype(bool)
    lines.append(f"Recordings with has_turn_data=False (no temporal features possible): {int(no_turn_data.sum())}\n")

    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0].sort_values(ascending=False)
    lines.append(f"Columns with at least one missing value: {len(na_counts)}\n")
    if len(na_counts):
        lines.append(na_counts.to_frame("n_missing").to_markdown())
        lines.append("")

    # incidental NaNs: missing in a row that DOES have turn data
    # (i.e. not explained by the has_turn_data=False block)
    incidental = df[df["has_turn_data"]].isna().sum()
    incidental = incidental[incidental > 0].sort_values(ascending=False)
    lines.append(f"Missing values among the has_turn_data=True rows only (incidental, not the expected block): {len(incidental)} columns affected\n")
    if len(incidental):
        lines.append(incidental.to_frame("n_missing").to_markdown())
        lines.append("")

    complete_case_mask = df["has_turn_data"].astype(bool) & df.notna().all(axis=1)
    n_dropped_for_incidental = int(df["has_turn_data"].sum() - complete_case_mask.sum())
    lines.append(
        f"Complete-case rows (has_turn_data=True AND no missing values anywhere): {int(complete_case_mask.sum())} "
        f"of {len(df)} total ({int(df['has_turn_data'].sum())} have turn data; "
        f"{n_dropped_for_incidental} of those dropped for incidental missing values elsewhere).\n"
    )
    lines.append(
        "Decision: the 95 has_turn_data=False recordings are NOT imputed — an entire feature "
        "family (temporal) would have to be fabricated for them, which is a bigger step than "
        "simple imputation. They stay in `modeling_dataset.csv` (with real NaNs, for future "
        "acoustic+MFCC-only modeling) but are excluded from the baseline modeling table "
        "(`modeling_dataset_complete_cases.csv`). Any remaining incidental NaNs (occasional "
        "F0/response-latency edge cases) are also handled by complete-case exclusion rather "
        "than imputation, since so few rows are affected — see the counts above.\n"
    )

    return "\n".join(lines), complete_case_mask


def run():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    combined, join_stats, families = build_combined_table()
    combined.to_csv(MODELING_DATASET_CSV, index=False)
    families.to_csv(FEATURE_FAMILIES_CSV, index=False)

    missing_md, complete_mask = missing_value_report(combined)
    complete_df = combined[complete_mask].copy()
    complete_df.to_csv(MODELING_DATASET_COMPLETE_CSV, index=False)

    # leakage check: train/val split never mixes the same anon_id, and
    # the split comes from manifest.csv (via the acoustic table), not
    # invented here
    train_ids = set(combined.loc[combined["split"] == "train", "anon_id"])
    val_ids = set(combined.loc[combined["split"] == "val", "anon_id"])
    overlap = train_ids & val_ids
    other_splits = sorted(set(combined["split"].unique()) - {"train", "val"})

    report = f"""# Modeling dataset preparation report

## Join
- Acoustic (recording_level.csv, version=original): {join_stats['n_acoustic']} recordings
- MFCC (mfcc_recording_level.csv, version=original): {join_stats['n_mfcc']} recordings
- Temporal (temporal_recording_level.csv): {join_stats['n_temporal']} recordings (the 258 with valid turn metadata)
- Combined table: {join_stats['n_combined']} rows, {combined.shape[1]} columns
- Row-count sanity: combined row count equals acoustic row count at every merge step (asserted in code) -> no duplicate-key row inflation
- Identity-column mismatches between acoustic and temporal tables (label/split/duration_s, where both present): {join_stats['identity_mismatches'] if join_stats['identity_mismatches'] else '(none)'}

{missing_md}

## Leakage check

- train anon_ids: {len(train_ids)}, val anon_ids: {len(val_ids)}
- train ∩ val: {len(overlap)} {'(EMPTY, as required)' if not overlap else '-- PROBLEM, non-empty!'}
- split values other than train/val found: {other_splits if other_splits else '(none)'}
- Split comes directly from `manifest.csv` (via the acoustic table) — not re-derived or randomized here.
- One row per recording/conversation throughout (no frame-level or turn-level rows in this table), so there is
  no risk of the same conversation contributing to both train and val through sub-observations.

## Output files

- `outputs/features/modeling_dataset.csv` — all {join_stats['n_combined']} recordings, real NaNs preserved
- `outputs/features/modeling_dataset_complete_cases.csv` — {int(complete_mask.sum())} recordings with every feature present (used by modeling/baseline.py)
"""
    with open(DATA_QUALITY_REPORT_MD, "w") as f:
        f.write(report)
    print(report)

    return {
        "n_combined": join_stats["n_combined"],
        "n_complete_cases": int(complete_mask.sum()),
        "leakage_overlap": len(overlap),
    }


if __name__ == "__main__":
    run()
