"""
ElevenLabs-fingerprint stage's audio adapter + feature extraction.
Reuses (does not duplicate) the resample/variant/feature-extraction
logic already established in external_data/adapt_and_extract_features.py
-- same 4 variants, same 8kHz target, same "no denoising, no compression,
no EQ beyond the already-established transforms" rule:

  - "original":             resample to 8000 Hz mono -> features
  - "normalized":           ... -> RMS-normalize -> features
  - "telephony":            ... -> band-limit 300-3400Hz -> features
  - "telephony_normalized": ... -> band-limit -> RMS-normalize -> features

Only two things are NEW here, both additive:
  1. Spectral flatness (preprocessing/features.py:extract_spectral_flatness),
     requested explicitly for this stage and not previously extracted
     anywhere in the project.
  2. crest_factor = peak_amplitude / rms, computed the same way
     modeling/diagnostics.py already derives it downstream of the
     acoustic table (not a new formula, just applied here too).

Human clips are ElevenLabs's 48kHz-native OpenSLR wavs reused as-is;
ElevenLabs clips are mp3 (44100Hz per the API's mp3_44100_128 output
format) -- both are decoded via soundfile and resampled to 8000 Hz here,
identically, so sample rate is never a source of difference between the
two groups downstream of this script.

Writes:
  external_data/elevenlabs/processed/<variant>/<anon_id>.wav
  outputs/features/elevenlabs_acoustic_<variant>.csv
  outputs/features/elevenlabs_mfcc_<variant>.csv
  outputs/features/elevenlabs_resample_log.csv
  outputs/features/elevenlabs_recording_level.csv   (long format: one row
    per anon_id x variant, joined with manifest metadata -- the single
    table modeling/elevenlabs_fingerprint_analysis.py consumes)
  Updates external_data/elevenlabs/manifest.csv with processed_path
  and outputs/reports/elevenlabs_manifest.csv (copy)

Run with: python external_data/adapt_and_extract_elevenlabs.py
  (after external_data/build_elevenlabs_manifest.py)
"""
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from preprocessing.audio_io import EXPECTED_SAMPLE_RATE, basic_signal_stats
from preprocessing.normalize import normalize_rms
from preprocessing.telephony_filter import apply_telephony_band
from preprocessing.features import (
    extract_f0, extract_rms, extract_spectral, extract_mfcc, mfcc_summary,
    nan_summary, extract_spectral_flatness,
)
from preprocessing.pipeline import FEATURES_DIR, REPORTS_DIR

EXTERNAL_DIR = Path(__file__).resolve().parent
ELEVENLABS_DIR = EXTERNAL_DIR / "elevenlabs"
PROCESSED_DIR = ELEVENLABS_DIR / "processed"
MANIFEST_CSV = ELEVENLABS_DIR / "manifest.csv"
MANIFEST_REPORT_COPY = REPORTS_DIR / "elevenlabs_manifest.csv"
RECORDING_LEVEL_CSV = FEATURES_DIR / "elevenlabs_recording_level.csv"

TARGET_SR = EXPECTED_SAMPLE_RATE  # 8000, same as the main pipeline
VARIANTS = ("original", "normalized", "telephony", "telephony_normalized")
# "native_rate" is a 5th, separately-handled condition: features extracted
# directly on the UN-resampled signal at its own native sample rate
# (48000Hz for OpenSLR, 44100Hz for ElevenLabs mp3_44100_128), specifically
# to test criterion 6E ("robust to sample-rate changes") -- not put through
# make_variant/normalize/telephony since those are only meaningful at 8kHz.


def load_native(path: Path) -> tuple:
    y, sr = sf.read(str(path), always_2d=True)
    y = y[:, 0].astype(np.float64)
    return y, sr


def load_and_resample(path: Path) -> tuple:
    y, sr = load_native(path)
    if sr != TARGET_SR:
        y = resample_poly(y, TARGET_SR, sr)
    return y, sr


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


def extract_row(y, sr, anon_id):
    stats = basic_signal_stats(y, sr)
    times, f0, voiced_flag, voiced_prob = extract_f0(y, sr)
    rms = extract_rms(y)
    centroid, bandwidth, rolloff, zcr = extract_spectral(y, sr)
    flatness = extract_spectral_flatness(y)
    n = min(len(times), len(rms), len(centroid), len(bandwidth), len(rolloff), len(zcr), len(flatness))
    voiced_proportion = float(sum(voiced_flag[:n])) / n if n else float("nan")

    peak = stats["peak_amplitude"]
    raw_rms = stats["rms"]
    crest_factor = float(peak / raw_rms) if raw_rms else float("nan")

    acoustic_row = {
        "anon_id": anon_id,
        "raw_peak_amplitude": peak,
        "raw_rms": raw_rms,
        "raw_silence_proportion": stats["silence_proportion"],
        "crest_factor": crest_factor,
    }
    acoustic_row.update(nan_summary(f0[:n], "f0"))
    acoustic_row["voiced_proportion"] = voiced_proportion
    acoustic_row.update(nan_summary(rms[:n], "rms"))
    acoustic_row.update(nan_summary(centroid[:n], "spectral_centroid"))
    acoustic_row.update(nan_summary(bandwidth[:n], "spectral_bandwidth"))
    acoustic_row.update(nan_summary(rolloff[:n], "spectral_rolloff"))
    acoustic_row.update(nan_summary(zcr[:n], "zcr"))
    acoustic_row.update(nan_summary(flatness[:n], "spectral_flatness"))

    mfcc = extract_mfcc(y, sr)
    mfcc_row = {"anon_id": anon_id}
    mfcc_row.update(mfcc_summary(mfcc))
    return acoustic_row, mfcc_row


def run():
    if not MANIFEST_CSV.exists():
        print(
            f"{MANIFEST_CSV} does not exist yet -- run "
            "external_data/build_elevenlabs_manifest.py first (which "
            "itself requires external_data/generate_elevenlabs.py to "
            "have produced generation_manifest.csv)."
        )
        sys.exit(1)

    for variant in VARIANTS:
        (PROCESSED_DIR / variant).mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_CSV, newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    all_conditions = VARIANTS + ("native_rate",)
    acoustic_rows = {v: [] for v in all_conditions}
    mfcc_rows = {v: [] for v in all_conditions}
    resample_log = []
    failures = []

    for row in manifest_rows:
        anon_id = row["anon_id"]
        raw_path = EXTERNAL_DIR / row["original_path"]
        try:
            y_native, orig_sr = load_native(raw_path)
            y_8k = resample_poly(y_native, TARGET_SR, orig_sr) if orig_sr != TARGET_SR else y_native
        except Exception as e:
            failures.append({"anon_id": anon_id, "reason": f"{type(e).__name__}: {e}"})
            print(f"FAILED {anon_id}: {e}")
            continue

        resample_log.append({"anon_id": anon_id, "original_sr": orig_sr, "target_sr": TARGET_SR})

        for variant in VARIANTS:
            y_variant = make_variant(y_8k, variant)
            out_path = PROCESSED_DIR / variant / f"{anon_id}.wav"
            sf.write(str(out_path), y_variant, TARGET_SR, subtype="PCM_16")
            acoustic_row, mfcc_row = extract_row(y_variant, TARGET_SR, anon_id)
            acoustic_rows[variant].append(acoustic_row)
            mfcc_rows[variant].append(mfcc_row)

        native_acoustic_row, native_mfcc_row = extract_row(y_native, orig_sr, anon_id)
        acoustic_rows["native_rate"].append(native_acoustic_row)
        mfcc_rows["native_rate"].append(native_mfcc_row)

        row["processed_path"] = str((PROCESSED_DIR / "original" / f"{anon_id}.wav").relative_to(EXTERNAL_DIR))
        print(f"processed {anon_id} (orig sr={orig_sr}Hz -> {TARGET_SR}Hz, 4 variants + native_rate)")

    for variant in all_conditions:
        if not acoustic_rows[variant]:
            continue
        with open(FEATURES_DIR / f"elevenlabs_acoustic_{variant}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(acoustic_rows[variant][0].keys()))
            w.writeheader()
            w.writerows(acoustic_rows[variant])
        with open(FEATURES_DIR / f"elevenlabs_mfcc_{variant}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(mfcc_rows[variant][0].keys()))
            w.writeheader()
            w.writerows(mfcc_rows[variant])

    with open(MANIFEST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)
    shutil.copy(MANIFEST_CSV, MANIFEST_REPORT_COPY)

    with open(FEATURES_DIR / "elevenlabs_resample_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["anon_id", "original_sr", "target_sr"])
        w.writeheader()
        w.writerows(resample_log)

    # Build the long-format joined recording-level table (one row per anon_id x variant)
    manifest_df = pd.read_csv(MANIFEST_CSV)
    long_rows = []
    for variant in all_conditions:
        if not acoustic_rows[variant]:
            continue
        acoustic_df = pd.DataFrame(acoustic_rows[variant])
        mfcc_df = pd.DataFrame(mfcc_rows[variant])
        merged = acoustic_df.merge(mfcc_df, on="anon_id", how="inner", validate="one_to_one")
        merged = merged.merge(
            manifest_df[["anon_id", "source", "label", "speaker_id", "voice_id",
                         "generator", "language", "text_id", "text"]],
            on="anon_id", how="left", validate="many_to_one",
        )
        merged.insert(1, "variant", variant)
        long_rows.append(merged)
    recording_level = pd.concat(long_rows, ignore_index=True)
    recording_level.to_csv(RECORDING_LEVEL_CSV, index=False)

    print(f"\nProcessed {len(resample_log)}/{len(manifest_rows)} clips x {len(VARIANTS)} variants. Failures: {len(failures)}")
    if failures:
        print(f"Failures: {failures}")
    print(f"Sample rates seen: {sorted(set(r['original_sr'] for r in resample_log))} -> {TARGET_SR} Hz")
    print(f"Wrote {RECORDING_LEVEL_CSV} ({len(recording_level)} rows)")


if __name__ == "__main__":
    run()
