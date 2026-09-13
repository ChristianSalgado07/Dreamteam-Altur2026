"""
Domain-matched experiment's audio adapter + feature extraction. Reuses
(does not duplicate) the resample/variant/feature-extraction logic from
external_data/adapt_and_extract_features.py — only the manifest and
output locations differ. Same rule as before: no denoising, compression,
EQ, or AGC; only resample-to-8kHz plus the already-established
normalize/telephony-filter transforms.

Writes:
  external_data/processed/domain_matched/<variant>/<external_id>.wav
  outputs/features/domain_matched_acoustic_<variant>.csv
  outputs/features/domain_matched_mfcc_<variant>.csv
  outputs/features/domain_matched_resample_log.csv
  Updates external_data/manifests/domain_matched_manifest.csv with processed_path
  and outputs/reports/domain_matched_manifest.csv (copy)

Run with: python external_data/adapt_and_extract_domain_matched.py
  (after build_domain_matched_manifest.py)

Note on FLAC decoding: a subset of the ASVspoof2021 FLAC files (~45% in
practice, not correlated with codec/label) trip a libsndfile decoding
bug on this system ("unknown error in flac decoder" / "lost sync"),
even though the files are valid FLAC (confirmed: `file` recognizes them
correctly and ffmpeg decodes them without issue). Rather than silently
dropping ~45% of the sample, `load_and_resample` is overridden here
(this script only) to decode via ffmpeg first — a format-compatibility
fix, not a data transformation: ffmpeg is told to only remux/decode to
PCM, no filters, no resampling at this stage (that still happens via
the same `resample_poly` step afterward).
"""
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from external_data.adapt_and_extract_features import (
    make_variant, extract_acoustic_and_mfcc, VARIANTS, TARGET_SR,
)
from preprocessing.pipeline import FEATURES_DIR, REPORTS_DIR


def load_and_resample(path: Path) -> tuple:
    """ffmpeg-decode (fixes the libsndfile FLAC issue above) -> mono float64 -> resample to 8kHz."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-ac", "1", tmp.name],
            check=True, capture_output=True,
        )
        y, sr = sf.read(tmp.name, always_2d=True)
    y = y[:, 0].astype(np.float64)
    if sr != TARGET_SR:
        y = resample_poly(y, TARGET_SR, sr)
    return y, sr

EXTERNAL_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXTERNAL_DIR / "processed" / "domain_matched"
MANIFEST_CSV = EXTERNAL_DIR / "manifests" / "domain_matched_manifest.csv"
MANIFEST_REPORT_COPY = REPORTS_DIR / "domain_matched_manifest.csv"


def run():
    for variant in VARIANTS:
        (PROCESSED_DIR / variant).mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_CSV, newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    acoustic_rows = {v: [] for v in VARIANTS}
    mfcc_rows = {v: [] for v in VARIANTS}
    resample_log = []
    failures = []

    for row in manifest_rows:
        external_id = row["external_id"]
        raw_path = EXTERNAL_DIR / row["original_path"]
        try:
            y_8k, orig_sr = load_and_resample(raw_path)
        except Exception as e:
            failures.append({"external_id": external_id, "reason": f"{type(e).__name__}: {e}"})
            print(f"FAILED {external_id}: {e}")
            continue

        resample_log.append({"external_id": external_id, "original_sr": orig_sr, "target_sr": TARGET_SR})

        for variant in VARIANTS:
            y_variant = make_variant(y_8k, variant)
            out_path = PROCESSED_DIR / variant / f"{external_id}.wav"
            sf.write(str(out_path), y_variant, TARGET_SR, subtype="PCM_16")

            acoustic_row, mfcc_row = extract_acoustic_and_mfcc(y_variant, TARGET_SR, external_id)
            acoustic_rows[variant].append(acoustic_row)
            mfcc_rows[variant].append(mfcc_row)

        row["processed_path"] = str((PROCESSED_DIR / "original" / f"{external_id}.wav").relative_to(EXTERNAL_DIR))
        print(f"processed {external_id} (orig sr={orig_sr}Hz -> {TARGET_SR}Hz, 4 variants)")

    for variant in VARIANTS:
        if not acoustic_rows[variant]:
            continue
        with open(FEATURES_DIR / f"domain_matched_acoustic_{variant}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(acoustic_rows[variant][0].keys()))
            w.writeheader()
            w.writerows(acoustic_rows[variant])
        with open(FEATURES_DIR / f"domain_matched_mfcc_{variant}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mfcc_rows[variant][0].keys()))
            w.writeheader()
            w.writerows(mfcc_rows[variant])

    with open(MANIFEST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)
    shutil.copy(MANIFEST_CSV, MANIFEST_REPORT_COPY)

    with open(FEATURES_DIR / "domain_matched_resample_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["external_id", "original_sr", "target_sr"])
        w.writeheader()
        w.writerows(resample_log)

    with open(REPORTS_DIR / "domain_matched_processing_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["external_id", "status", "reason"])
        processed_ids = {r["external_id"] for r in resample_log}
        for row in manifest_rows:
            if row["external_id"] in processed_ids:
                w.writerow([row["external_id"], "ok", ""])
        for fail in failures:
            w.writerow([fail["external_id"], "failed", fail["reason"]])

    print(f"\nProcessed {len(resample_log)}/{len(manifest_rows)} clips x {len(VARIANTS)} variants. Failures: {len(failures)}")
    if failures:
        print(f"Failures: {failures}")
    print(f"Sample rates seen: {sorted(set(r['original_sr'] for r in resample_log))} -> {TARGET_SR} Hz")


if __name__ == "__main__":
    run()
