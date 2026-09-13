"""
Acoustic + MFCC feature extraction on TELEPHONY-BAND-LIMITED Channel-0
audio — a new, additive experiment path (preprocessing/telephony_filter.py).
Does not touch the original/denoised pipeline (preprocessing/pipeline.py)
or the RMS-normalization experiment (preprocessing/normalized_pipeline.py).

Two variants per recording, computed from ONE filtering pass (the
normalization step is applied afterward, as an explicit second
operation, never fused into the filter itself):

  A. "telephony"            = original -> band-limit -> features
  B. "telephony_normalized" = original -> band-limit -> RMS-normalize -> features

Reuses the SAME feature-extraction functions, frame settings, and
naming conventions as preprocessing/features.py — only the input
signal differs.

Writes:
  outputs/features/telephony_acoustic.csv
  outputs/features/telephony_mfcc.csv
  outputs/features/telephony_normalized_acoustic.csv
  outputs/features/telephony_normalized_mfcc.csv

Run with: python -m preprocessing.telephony_pipeline
"""
import csv
import multiprocessing as mp
import time

from preprocessing.audio_io import AUDIO_DIR, load_channel0, basic_signal_stats
from preprocessing.telephony_filter import apply_telephony_band, LOW_CUTOFF_HZ, HIGH_CUTOFF_HZ, ORDER
from preprocessing.normalize import normalize_rms, TARGET_RMS
from preprocessing.features import extract_f0, extract_rms, extract_spectral, extract_mfcc, mfcc_summary, nan_summary
from preprocessing.pipeline import FEATURES_DIR, REPORTS_DIR

TELEPHONY_ACOUSTIC_CSV = FEATURES_DIR / "telephony_acoustic.csv"
TELEPHONY_MFCC_CSV = FEATURES_DIR / "telephony_mfcc.csv"
TELEPHONY_NORMALIZED_ACOUSTIC_CSV = FEATURES_DIR / "telephony_normalized_acoustic.csv"
TELEPHONY_NORMALIZED_MFCC_CSV = FEATURES_DIR / "telephony_normalized_mfcc.csv"
TELEPHONY_REPORT_MD = REPORTS_DIR / "telephony_processing_report.md"


def _extract_acoustic_and_mfcc(y, sr, anon_id, version_label):
    stats = basic_signal_stats(y, sr)
    times, f0, voiced_flag, voiced_prob = extract_f0(y, sr)
    rms = extract_rms(y)
    centroid, bandwidth, rolloff, zcr = extract_spectral(y, sr)
    n = min(len(times), len(rms), len(centroid), len(bandwidth), len(rolloff), len(zcr))
    voiced_proportion = float(sum(voiced_flag[:n])) / n if n else float("nan")

    acoustic_row = {
        "anon_id": anon_id, "version": version_label,
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
    mfcc_row = {"anon_id": anon_id, "version": version_label}
    mfcc_row.update(mfcc_summary(mfcc))
    return acoustic_row, mfcc_row


def process_one(anon_id: str):
    try:
        y, sr = load_channel0(anon_id)
        y_band, filter_params = apply_telephony_band(y, sr)

        acoustic_a, mfcc_a = _extract_acoustic_and_mfcc(y_band, sr, anon_id, "telephony")

        y_band_norm, gain_info = normalize_rms(y_band)
        acoustic_b, mfcc_b = _extract_acoustic_and_mfcc(y_band_norm, sr, anon_id, "telephony_normalized")

        return {
            "status": "ok", "anon_id": anon_id,
            "acoustic_a": acoustic_a, "mfcc_a": mfcc_a,
            "acoustic_b": acoustic_b, "mfcc_b": mfcc_b,
        }
    except Exception as e:
        return {"status": "failed", "anon_id": anon_id, "reason": f"{type(e).__name__}: {e}"}


def run(n_workers: int = 6, limit: int = None):
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    audio_ids = sorted(p.stem for p in AUDIO_DIR.glob("*.wav"))
    if limit:
        audio_ids = audio_ids[:limit]

    t0 = time.time()
    failures = []
    n_ok = 0
    writers = {}

    files = {
        "acoustic_a": open(TELEPHONY_ACOUSTIC_CSV, "w", newline=""),
        "mfcc_a": open(TELEPHONY_MFCC_CSV, "w", newline=""),
        "acoustic_b": open(TELEPHONY_NORMALIZED_ACOUSTIC_CSV, "w", newline=""),
        "mfcc_b": open(TELEPHONY_NORMALIZED_MFCC_CSV, "w", newline=""),
    }
    try:
        with mp.Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(process_one, audio_ids), start=1):
                if result["status"] != "ok":
                    failures.append(result)
                    print(f"[{i}/{len(audio_ids)}] FAILED {result['anon_id']}: {result['reason']}", flush=True)
                    continue

                for key in ("acoustic_a", "mfcc_a", "acoustic_b", "mfcc_b"):
                    row = result[key]
                    if key not in writers:
                        writers[key] = csv.DictWriter(files[key], fieldnames=list(row.keys()))
                        writers[key].writeheader()
                    writers[key].writerow(row)
                    files[key].flush()
                n_ok += 1
                print(f"[{i}/{len(audio_ids)}] ok {result['anon_id']}", flush=True)
    finally:
        for f in files.values():
            f.close()

    elapsed = time.time() - t0
    report = f"""# Telephony-band processing report

- Recordings expected: {len(audio_ids)}
- Successfully processed: {n_ok}
- Failed: {len(failures)}
- Wall time: {elapsed:.1f}s ({n_workers} worker processes)

## Filter settings

- Method: Butterworth band-pass, order {ORDER}, second-order-sections, zero-phase (sosfiltfilt)
- Cutoffs: {LOW_CUTOFF_HZ} Hz - {HIGH_CUTOFF_HZ} Hz (standard analog telephony voice band)
- Same filter for every recording, independent of label
- No resampling, no denoising, no compression, no EQ beyond this band limit

## Two variants produced

- A "telephony": original -> band-limit -> features (telephony_acoustic.csv, telephony_mfcc.csv)
- B "telephony_normalized": original -> band-limit -> RMS-normalize (target {TARGET_RMS}) -> features
  (telephony_normalized_acoustic.csv, telephony_normalized_mfcc.csv)

## Failures

{chr(10).join(f"- {r['anon_id']}: {r['reason']}" for r in failures) if failures else "(none)"}
"""
    with open(TELEPHONY_REPORT_MD, "w") as f:
        f.write(report)
    print(report)
    return {"ok": n_ok, "failed": len(failures), "elapsed_s": elapsed}


if __name__ == "__main__":
    run()
