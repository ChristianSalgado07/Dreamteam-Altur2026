"""
Builds the level-normalization experiment's modeling table: RMS-
normalized acoustic + MFCC features, joined with the EXISTING (not
recomputed) temporal features and the existing acoustic table's
identity columns (label/split/duration_s/has_turn_data — these are
recording metadata, independent of which signal-processing version of
the audio produced the acoustic features, so they're reused rather than
duplicated).

Temporal features are reused as-is: within_turn_pause_* is the only
temporal feature family that depends on signal amplitude at all (an
RMS-percentile threshold within each recording's own target turns —
see temporal/turn_features.py), and that threshold is SCALE-INVARIANT
under a per-recording positive linear gain (RMS normalization is
exactly that): scaling every sample by a constant scales the threshold
by the same constant, so which frames fall below it is unchanged. This
is stated explicitly here rather than silently reusing the values.

Does not modify modeling/prepare_dataset.py or any of its outputs.

Run with: python -m modeling.prepare_dataset_normalized
"""
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.pipeline import FEATURES_DIR as ACOUSTIC_FEATURES_DIR
from modeling.prepare_dataset import FEATURES_DIR, REPORTS_DIR, load_temporal

ACOUSTIC_ORIGINAL_CSV = ACOUSTIC_FEATURES_DIR / "recording_level.csv"
RECORDING_LEVEL_NORMALIZED_CSV = FEATURES_DIR / "recording_level_normalized.csv"
MFCC_NORMALIZED_CSV = FEATURES_DIR / "mfcc_normalized.csv"

MODELING_DATASET_NORMALIZED_CSV = FEATURES_DIR / "modeling_dataset_normalized.csv"
MODELING_DATASET_NORMALIZED_COMPLETE_CSV = FEATURES_DIR / "modeling_dataset_normalized_complete_cases.csv"
NORMALIZED_DATA_QUALITY_REPORT_MD = REPORTS_DIR / "modeling_data_quality_report_normalized.md"


def load_identity_columns() -> pd.DataFrame:
    """anon_id, label, split, duration_s, has_turn_data — from the
    EXISTING original acoustic table (these don't depend on which audio
    version produced the acoustic features)."""
    df = pd.read_csv(ACOUSTIC_ORIGINAL_CSV)
    df = df[df["version"] == "original"][["anon_id", "label", "split", "duration_s", "has_turn_data"]]
    assert df["anon_id"].is_unique
    return df


def build_combined_table() -> tuple:
    identity = load_identity_columns()
    acoustic_norm = pd.read_csv(RECORDING_LEVEL_NORMALIZED_CSV).drop(columns=["version"])
    mfcc_norm = pd.read_csv(MFCC_NORMALIZED_CSV).drop(columns=["version"])
    temporal = load_temporal()

    n_identity, n_acoustic, n_mfcc, n_temporal = len(identity), len(acoustic_norm), len(mfcc_norm), len(temporal)
    for name, d in (("identity", identity), ("acoustic_norm", acoustic_norm), ("mfcc_norm", mfcc_norm), ("temporal", temporal)):
        assert d["anon_id"].is_unique, f"duplicate anon_id in {name}"

    combined = identity.merge(acoustic_norm, on="anon_id", how="left", validate="one_to_one")
    assert len(combined) == n_identity, "identity-acoustic merge changed row count"
    combined = combined.merge(mfcc_norm, on="anon_id", how="left", validate="one_to_one")
    assert len(combined) == n_identity, "acoustic-mfcc merge changed row count"

    check_cols = [c for c in ("label", "split", "duration_s", "has_turn_data") if c in temporal.columns]
    temporal_features_only = temporal.drop(columns=check_cols)
    combined = combined.merge(temporal_features_only, on="anon_id", how="left", validate="one_to_one")
    assert len(combined) == n_identity, "acoustic-temporal merge changed row count"

    return combined, {
        "n_identity": n_identity, "n_acoustic": n_acoustic, "n_mfcc": n_mfcc, "n_temporal": n_temporal,
        "n_combined": len(combined),
    }


def verification_checks(df: pd.DataFrame, complete_mask: pd.Series) -> str:
    lines = ["## Verification\n"]

    lines.append(f"- Recordings expected: 353. Present: {len(df)}. {'✅' if len(df) == 353 else '❌ MISMATCH'}")
    label_counts = df["label"].value_counts()
    n_synth = int(label_counts.get("synthetic", 0))
    n_human = int(label_counts.get("human", 0))
    lines.append(f"- Label counts: synthetic={n_synth} (expected 203), human={n_human} (expected 150). "
                 f"{'✅' if (n_synth, n_human) == (203, 150) else '❌ MISMATCH'}")

    # cross-check against the ORIGINAL dataset's labels/splits (same manifest.csv source) —
    # confirms no accidental label change or recording drop while building this new table
    orig = pd.read_csv(ACOUSTIC_ORIGINAL_CSV)
    orig = orig[orig["version"] == "original"][["anon_id", "label", "split"]]
    merged = df[["anon_id", "label", "split"]].merge(orig, on="anon_id", suffixes=("_norm", "_orig"))
    label_mismatches = int((merged["label_norm"] != merged["label_orig"]).sum())
    split_mismatches = int((merged["split_norm"] != merged["split_orig"]).sum())
    lines.append(f"- Label matches original dataset for every anon_id: {len(merged) - label_mismatches}/{len(merged)} "
                 f"({'✅' if label_mismatches == 0 else '❌ ' + str(label_mismatches) + ' MISMATCHES'})")
    lines.append(f"- Split assignment matches original dataset for every anon_id: {len(merged) - split_mismatches}/{len(merged)} "
                 f"({'✅' if split_mismatches == 0 else '❌ ' + str(split_mismatches) + ' MISMATCHES'})")

    train_ids = set(df.loc[df["split"] == "train", "anon_id"])
    val_ids = set(df.loc[df["split"] == "val", "anon_id"])
    overlap = train_ids & val_ids
    lines.append(f"- train ∩ val: {len(overlap)} ({'✅ empty' if not overlap else '❌ NON-EMPTY'})")

    dup_count = int(df["anon_id"].duplicated().sum())
    lines.append(f"- Duplicate anon_id rows: {dup_count} ({'✅' if dup_count == 0 else '❌'})")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_inf = int(np.isinf(df[numeric_cols].to_numpy()).sum())
    lines.append(f"- Infinite values: {n_inf} ({'✅' if n_inf == 0 else '❌'})")

    na_counts = df.isna().sum()
    na_counts = na_counts[na_counts > 0].sort_values(ascending=False)
    lines.append(f"- Columns with any missing value: {len(na_counts)} "
                 f"(expected: only the temporal-feature columns, for the 95 has_turn_data=False recordings)\n")
    if len(na_counts):
        lines.append(na_counts.to_frame("n_missing").to_markdown())
        lines.append("")

    incidental = df[df["has_turn_data"]].isna().sum()
    incidental = incidental[incidental > 0]
    lines.append(f"- Incidental missing values among has_turn_data=True rows: {len(incidental)} columns affected "
                 f"({'✅ none' if len(incidental) == 0 else '❌ see above'})\n")

    lines.append(f"- Complete-case rows (has_turn_data=True, no missing values): {int(complete_mask.sum())}\n")
    lines.append(
        "`duration_s` is present in the saved CSV for reference but is NOT used as a model "
        "feature (excluded via the same `NON_FEATURE_COLS` list used for the original dataset).\n"
    )
    return "\n".join(lines)


def run():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    combined, join_stats = build_combined_table()
    combined.to_csv(MODELING_DATASET_NORMALIZED_CSV, index=False)

    complete_mask = combined["has_turn_data"].astype(bool) & combined.notna().all(axis=1)
    complete_df = combined[complete_mask].copy()
    complete_df.to_csv(MODELING_DATASET_NORMALIZED_COMPLETE_CSV, index=False)

    verification_md = verification_checks(combined, complete_mask)

    report = f"""# Normalized modeling dataset preparation report

## Join
- Identity columns (label/split/duration_s/has_turn_data, from the EXISTING original acoustic table): {join_stats['n_identity']} recordings
- Normalized acoustic (recording_level_normalized.csv): {join_stats['n_acoustic']} recordings
- Normalized MFCC (mfcc_normalized.csv): {join_stats['n_mfcc']} recordings
- Temporal (EXISTING temporal_recording_level.csv, reused unchanged — see module docstring for why this is valid): {join_stats['n_temporal']} recordings
- Combined table: {join_stats['n_combined']} rows, {combined.shape[1]} columns

{verification_md}

## Output files

- `outputs/features/modeling_dataset_normalized.csv` — all {join_stats['n_combined']} recordings
- `outputs/features/modeling_dataset_normalized_complete_cases.csv` — {int(complete_mask.sum())} recordings (used by the normalized evaluation)
"""
    with open(NORMALIZED_DATA_QUALITY_REPORT_MD, "w") as f:
        f.write(report)
    print(report)

    return {"n_combined": join_stats["n_combined"], "n_complete_cases": int(complete_mask.sum())}


if __name__ == "__main__":
    run()
