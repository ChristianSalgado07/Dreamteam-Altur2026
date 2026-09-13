"""
Builds the telephony-band experiment's two modeling tables (variant A:
band-limited only; variant B: band-limited + RMS-normalized), joined
with the EXISTING (not recomputed) temporal features and the existing
acoustic table's identity columns — same pattern as
modeling/prepare_dataset_normalized.py.

Temporal features are reused as-is for the same reason documented in
prepare_dataset_normalized.py: within_turn_pause_* is the only
amplitude-dependent temporal family, and it depends on an RMS
PERCENTILE threshold computed within each recording's own target turns
— invariant to any monotonic, per-recording-consistent transform of the
RMS scale. A band-pass filter changes the absolute RMS values (unlike a
pure gain), but does not change the RANK ORDER of RMS across a
recording's own frames in a way that would flip which frames fall in
the bottom 20th percentile for THIS specific recording, since the same
filter is applied uniformly across time within one recording. This is
a weaker invariance guarantee than the pure-gain case (RMS-normalization
experiment), so it is stated explicitly here as an assumption, not
re-derived from scratch, rather than silently reused without comment.

Does not modify modeling/prepare_dataset.py, modeling/prepare_dataset_normalized.py,
or any of their outputs.

Run with: python -m modeling.prepare_dataset_telephony
"""
import numpy as np
import pandas as pd

from preprocessing.pipeline import FEATURES_DIR as ACOUSTIC_FEATURES_DIR
from modeling.prepare_dataset import FEATURES_DIR, REPORTS_DIR, load_temporal

ACOUSTIC_ORIGINAL_CSV = ACOUSTIC_FEATURES_DIR / "recording_level.csv"

VARIANTS = {
    "telephony": {
        "acoustic_csv": FEATURES_DIR / "telephony_acoustic.csv",
        "mfcc_csv": FEATURES_DIR / "telephony_mfcc.csv",
        "dataset_csv": FEATURES_DIR / "modeling_dataset_telephony.csv",
        "complete_csv": FEATURES_DIR / "modeling_dataset_telephony_complete_cases.csv",
        "report_md": REPORTS_DIR / "modeling_data_quality_report_telephony.md",
    },
    "telephony_normalized": {
        "acoustic_csv": FEATURES_DIR / "telephony_normalized_acoustic.csv",
        "mfcc_csv": FEATURES_DIR / "telephony_normalized_mfcc.csv",
        "dataset_csv": FEATURES_DIR / "modeling_dataset_telephony_normalized.csv",
        "complete_csv": FEATURES_DIR / "modeling_dataset_telephony_normalized_complete_cases.csv",
        "report_md": REPORTS_DIR / "modeling_data_quality_report_telephony_normalized.md",
    },
}


def load_identity_columns() -> pd.DataFrame:
    df = pd.read_csv(ACOUSTIC_ORIGINAL_CSV)
    df = df[df["version"] == "original"][["anon_id", "label", "split", "duration_s", "has_turn_data"]]
    assert df["anon_id"].is_unique
    return df


def build_combined_table(acoustic_csv, mfcc_csv) -> tuple:
    identity = load_identity_columns()
    acoustic = pd.read_csv(acoustic_csv).drop(columns=["version"])
    mfcc = pd.read_csv(mfcc_csv).drop(columns=["version"])
    temporal = load_temporal()

    n_identity = len(identity)
    for name, d in (("identity", identity), ("acoustic", acoustic), ("mfcc", mfcc), ("temporal", temporal)):
        assert d["anon_id"].is_unique, f"duplicate anon_id in {name}"

    combined = identity.merge(acoustic, on="anon_id", how="left", validate="one_to_one")
    assert len(combined) == n_identity
    combined = combined.merge(mfcc, on="anon_id", how="left", validate="one_to_one")
    assert len(combined) == n_identity

    check_cols = [c for c in ("label", "split", "duration_s", "has_turn_data") if c in temporal.columns]
    combined = combined.merge(temporal.drop(columns=check_cols), on="anon_id", how="left", validate="one_to_one")
    assert len(combined) == n_identity

    return combined, {"n_identity": n_identity, "n_acoustic": len(acoustic), "n_mfcc": len(mfcc), "n_temporal": len(temporal)}


def verification_checks(df: pd.DataFrame, complete_mask: pd.Series) -> str:
    lines = ["## Verification\n"]
    lines.append(f"- Recordings expected: 353. Present: {len(df)}. {'✅' if len(df) == 353 else '❌ MISMATCH'}")
    label_counts = df["label"].value_counts()
    n_synth, n_human = int(label_counts.get("synthetic", 0)), int(label_counts.get("human", 0))
    lines.append(f"- Label counts: synthetic={n_synth} (expected 203), human={n_human} (expected 150). "
                 f"{'✅' if (n_synth, n_human) == (203, 150) else '❌ MISMATCH'}")

    orig = pd.read_csv(ACOUSTIC_ORIGINAL_CSV)
    orig = orig[orig["version"] == "original"][["anon_id", "label", "split"]]
    merged = df[["anon_id", "label", "split"]].merge(orig, on="anon_id", suffixes=("_this", "_orig"))
    label_mismatches = int((merged["label_this"] != merged["label_orig"]).sum())
    split_mismatches = int((merged["split_this"] != merged["split_orig"]).sum())
    lines.append(f"- Label matches original dataset for every anon_id: {'✅' if label_mismatches == 0 else f'❌ {label_mismatches} MISMATCHES'}")
    lines.append(f"- Split matches original dataset for every anon_id: {'✅' if split_mismatches == 0 else f'❌ {split_mismatches} MISMATCHES'}")

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
    na_counts = na_counts[na_counts > 0]
    incidental = df[df["has_turn_data"]].isna().sum()
    incidental = incidental[incidental > 0]
    lines.append(f"- Columns with any missing value: {len(na_counts)} (expected: temporal columns for the 95 has_turn_data=False recordings)")
    lines.append(f"- Incidental missing values among has_turn_data=True rows: {len(incidental)} ({'✅ none' if len(incidental) == 0 else '❌ see data'})")
    lines.append(f"- Complete-case rows (has_turn_data=True, no missing values): {int(complete_mask.sum())}\n")
    lines.append("`duration_s` is present for reference but NOT used as a model feature.\n")
    return "\n".join(lines)


def build_one(variant_name: str):
    v = VARIANTS[variant_name]
    combined, join_stats = build_combined_table(v["acoustic_csv"], v["mfcc_csv"])
    combined.to_csv(v["dataset_csv"], index=False)

    complete_mask = combined["has_turn_data"].astype(bool) & combined.notna().all(axis=1)
    combined[complete_mask].to_csv(v["complete_csv"], index=False)

    verification_md = verification_checks(combined, complete_mask)
    report = f"""# Telephony modeling dataset preparation report — variant "{variant_name}"

## Join
- Identity columns (from the EXISTING original acoustic table): {join_stats['n_identity']} recordings
- Acoustic ({v['acoustic_csv'].name}): {join_stats['n_acoustic']} recordings
- MFCC ({v['mfcc_csv'].name}): {join_stats['n_mfcc']} recordings
- Temporal (EXISTING temporal_recording_level.csv, reused — see module docstring): {join_stats['n_temporal']} recordings
- Combined table: {len(combined)} rows, {combined.shape[1]} columns

{verification_md}

## Output files
- `{v['dataset_csv'].name}` — all {len(combined)} recordings
- `{v['complete_csv'].name}` — {int(complete_mask.sum())} recordings (complete cases)
"""
    with open(v["report_md"], "w") as f:
        f.write(report)
    print(report)
    return {"n_combined": len(combined), "n_complete_cases": int(complete_mask.sum())}


def run():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant_name in VARIANTS:
        print(f"=== Building variant: {variant_name} ===")
        results[variant_name] = build_one(variant_name)
    return results


if __name__ == "__main__":
    run()
