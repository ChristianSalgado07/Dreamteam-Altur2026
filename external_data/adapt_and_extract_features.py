"""
External-audio adapter + feature extraction. The ONLY preprocessing
applied here is what's needed to make external audio consistent with
the existing pipeline's assumptions (8000 Hz, mono) plus, for 3 of the
4 variants, the SAME already-defined experimental transforms used on
the main dataset — nothing new, nothing silent:

  - "original":             resample to 8000 Hz mono -> features
  - "normalized":           ... -> RMS-normalize (preprocessing/normalize.py) -> features
  - "telephony":            ... -> band-limit 300-3400Hz (preprocessing/telephony_filter.py) -> features
  - "telephony_normalized": ... -> band-limit -> RMS-normalize -> features

No denoising, no compression, no EQ beyond the telephony variant's
already-documented band limit, no automatic gain control. Resampling
uses scipy.signal.resample_poly (same tool already used elsewhere in
this project, e.g. nlp/asr.py) — original and target sample rates are
recorded per clip in the manifest (already captured) and documented
here (48000/24000/22050 -> 8000 Hz, exact ratios below).

Only ACOUSTIC + MFCC features are extracted — NOT temporal features.
Temporal features (turn duration, response latency, turn-taking) are
structurally inapplicable: these are isolated single-speaker utterances
with no conversational partner or turn boundaries, unlike the main
dataset's multi-turn phone calls. This is a deliberate, documented scope
limitation, not a silent omission — see external_data/README.md and
outputs/reports/external_validation_report.md.

Writes:
  external_data/processed/<variant>/<external_id>.wav   (4 variants)
  outputs/features/external_acoustic_<variant>.csv
  outputs/features/external_mfcc_<variant>.csv
  Updates external_data/manifests/external_manifest.csv with processed_path
  (the "original" variant's path, as the canonical 8kHz reference)
  and outputs/reports/external_manifest.csv (a copy, per the task's requested location)

Run with: python external_data/adapt_and_extract_features.py
"""
import csv
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from preprocessing.audio_io import EXPECTED_SAMPLE_RATE, basic_signal_stats
from preprocessing.normalize import normalize_rms
from preprocessing.telephony_filter import apply_telephony_band
from preprocessing.features import extract_f0, extract_rms, extract_spectral, extract_mfcc, mfcc_summary, nan_summary
from preprocessing.pipeline import FEATURES_DIR, REPORTS_DIR

EXTERNAL_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXTERNAL_DIR / "processed"
MANIFEST_CSV = EXTERNAL_DIR / "manifests" / "external_manifest.csv"
EXTERNAL_MANIFEST_REPORT_COPY = REPORTS_DIR / "external_manifest.csv"

TARGET_SR = EXPECTED_SAMPLE_RATE  # 8000, same as the main pipeline

VARIANTS = ("original", "normalized", "telephony", "telephony_normalized")


def load_and_resample(path: Path) -> tuple:
    y, sr = sf.read(str(path), always_2d=True)
    y = y[:, 0].astype(np.float64)  # these sources are already mono; take channel 0 defensively
    if sr != TARGET_SR:
        y = resample_poly(y, TARGET_SR, sr)
    return y, sr  # sr = ORIGINAL sample rate, for documentation


def make_variant(y_8k: np.ndarray, variant: str) -> np.ndarray:
    if variant == "original":
        return y_8k
    if variant == "normalized":
        y_norm, _ = normalize_rms(y_8k)
        return y_norm
    if variant == "telephony":
        y_band, _ = apply_telephony_band(y_8k, TARGET_SR)
        return y_band
    if variant == "telephony_normalized":
        y_band, _ = apply_telephony_band(y_8k, TARGET_SR)
        y_band_norm, _ = normalize_rms(y_band)
        return y_band_norm
    raise ValueError(variant)


def extract_acoustic_and_mfcc(y, sr, external_id):
    stats = basic_signal_stats(y, sr)
    times, f0, voiced_flag, voiced_prob = extract_f0(y, sr)
    rms = extract_rms(y)
    centroid, bandwidth, rolloff, zcr = extract_spectral(y, sr)
    n = min(len(times), len(rms), len(centroid), len(bandwidth), len(rolloff), len(zcr))
    voiced_proportion = float(sum(voiced_flag[:n])) / n if n else float("nan")

    acoustic_row = {
        "external_id": external_id,
        "raw_peak_amplitude": stats["peak_amplitude"],
        "raw_rms": stats["rms"],
        "raw_silence_proportion": stats["silence_proportion"],
    }
    acoustic_row.update(nan_summary(f0[:n], "f0"))
    acoustic_row["voiced_proportion"] = voiced_proportion
    acoustic_row.update(nan_summary(rms[:n], "rms"))
    acoustic_row.update(nan_summary(centroid[:n], "spectral_centroid"))
    acoustic_row.update(nan_summary(bandwidth[:n], "spectral_bandwidth"))
    acoustic_row.update(nan_summary(rolloff[:n], "spectral_rolloff"))
    acoustic_row.update(nan_summary(zcr[:n], "zcr"))

    mfcc = extract_mfcc(y, sr)
    mfcc_row = {"external_id": external_id}
    mfcc_row.update(mfcc_summary(mfcc))
    return acoustic_row, mfcc_row


def run():
    for variant in VARIANTS:
        (PROCESSED_DIR / variant).mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_CSV, newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    acoustic_rows = {v: [] for v in VARIANTS}
    mfcc_rows = {v: [] for v in VARIANTS}
    resample_log = []

    for row in manifest_rows:
        external_id = row["external_id"]
        raw_path = EXTERNAL_DIR / row["original_path"]
        y_8k, orig_sr = load_and_resample(raw_path)
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
        with open(FEATURES_DIR / f"external_acoustic_{variant}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(acoustic_rows[variant][0].keys()))
            w.writeheader()
            w.writerows(acoustic_rows[variant])
        with open(FEATURES_DIR / f"external_mfcc_{variant}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mfcc_rows[variant][0].keys()))
            w.writeheader()
            w.writerows(mfcc_rows[variant])

    with open(MANIFEST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)
    shutil.copy(MANIFEST_CSV, EXTERNAL_MANIFEST_REPORT_COPY)

    resample_csv = FEATURES_DIR / "external_resample_log.csv"
    with open(resample_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["external_id", "original_sr", "target_sr"])
        w.writeheader()
        w.writerows(resample_log)

    print(f"\nProcessed {len(manifest_rows)} clips x {len(VARIANTS)} variants.")
    print(f"Sample rates seen: {sorted(set(r['original_sr'] for r in resample_log))} -> {TARGET_SR} Hz")
    print(f"Manifest updated: {MANIFEST_CSV}")
    print(f"Copy for reports: {EXTERNAL_MANIFEST_REPORT_COPY}")


if __name__ == "__main__":
    run()
