"""
Acoustic + MFCC feature extraction on RMS-LEVEL-NORMALIZED Channel-0
audio — a new, additive experiment path (preprocessing/normalize.py).
Does not touch or rerun the existing original/denoised acoustic
pipeline (preprocessing/pipeline.py, preprocessing/mfcc_pipeline.py) —
both of those outputs are untouched.

Reuses the SAME feature-extraction functions, frame settings, and
naming conventions as preprocessing/features.py — only the input
signal differs (RMS-normalized instead of original/denoised) — so
normalized and original feature tables have identical columns and can
be compared directly.

Writes:
  outputs/features/recording_level_normalized.csv  (F0/RMS/spectral/ZCR)
  outputs/features/mfcc_normalized.csv             (MFCC 1-13 mean/std)
  outputs/features/normalization_gains.csv         (per-recording gain/clip info)

Run with: python -m preprocessing.normalized_pipeline
"""
import csv
import multiprocessing as mp
import time

from preprocessing.audio_io import AUDIO_DIR, load_channel0, basic_signal_stats
from preprocessing.normalize import normalize_rms, TARGET_RMS, PEAK_SAFETY
from preprocessing.features import extract_f0, extract_rms, extract_spectral, extract_mfcc, mfcc_summary, nan_summary
from preprocessing.pipeline import FEATURES_DIR, REPORTS_DIR

RECORDING_LEVEL_NORMALIZED_CSV = FEATURES_DIR / "recording_level_normalized.csv"
MFCC_NORMALIZED_CSV = FEATURES_DIR / "mfcc_normalized.csv"
GAINS_CSV = FEATURES_DIR / "normalization_gains.csv"
NORMALIZED_REPORT_MD = REPORTS_DIR / "normalization_processing_report.md"


def process_one(anon_id: str):
    try:
        y, sr = load_channel0(anon_id)
        y_norm, gain_info = normalize_rms(y)

        # basic signal stats on the NORMALIZED signal (same function used
        # for the original/denoised paths, just fed different audio)
        stats = basic_signal_stats(y_norm, sr)

        times, f0, voiced_flag, voiced_prob = extract_f0(y_norm, sr)
        rms = extract_rms(y_norm)
        centroid, bandwidth, rolloff, zcr = extract_spectral(y_norm, sr)
        n = min(len(times), len(rms), len(centroid), len(bandwidth), len(rolloff), len(zcr))
        voiced_proportion = float(sum(voiced_flag[:n])) / n if n else float("nan")

        acoustic_row = {
            "anon_id": anon_id, "version": "normalized",
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

        mfcc = extract_mfcc(y_norm, sr)
        mfcc_row = {"anon_id": anon_id, "version": "normalized"}
        mfcc_row.update(mfcc_summary(mfcc))

        gain_row = {
            "anon_id": anon_id,
            "original_rms": gain_info["current_rms"],
            "original_peak_amplitude": gain_info["peak_amplitude"],
            "gain_applied": gain_info["gain"],
            "peak_capped": gain_info["peak_capped"],
            "degenerate_signal": gain_info["degenerate_signal"],
            "normalized_rms": stats["rms"],
            "normalized_peak_amplitude": stats["peak_amplitude"],
        }

        return {"status": "ok", "anon_id": anon_id, "acoustic_row": acoustic_row,
                "mfcc_row": mfcc_row, "gain_row": gain_row}
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
    acoustic_writer = mfcc_writer = gain_writer = None

    with open(RECORDING_LEVEL_NORMALIZED_CSV, "w", newline="") as f_ac, \
         open(MFCC_NORMALIZED_CSV, "w", newline="") as f_mfcc, \
         open(GAINS_CSV, "w", newline="") as f_gain, \
         mp.Pool(processes=n_workers) as pool:

        for i, result in enumerate(pool.imap_unordered(process_one, audio_ids), start=1):
            if result["status"] != "ok":
                failures.append(result)
                print(f"[{i}/{len(audio_ids)}] FAILED {result['anon_id']}: {result['reason']}", flush=True)
                continue

            if acoustic_writer is None:
                acoustic_writer = csv.DictWriter(f_ac, fieldnames=list(result["acoustic_row"].keys()))
                acoustic_writer.writeheader()
                mfcc_writer = csv.DictWriter(f_mfcc, fieldnames=list(result["mfcc_row"].keys()))
                mfcc_writer.writeheader()
                gain_writer = csv.DictWriter(f_gain, fieldnames=list(result["gain_row"].keys()))
                gain_writer.writeheader()

            acoustic_writer.writerow(result["acoustic_row"])
            mfcc_writer.writerow(result["mfcc_row"])
            gain_writer.writerow(result["gain_row"])
            f_ac.flush()
            f_mfcc.flush()
            f_gain.flush()
            n_ok += 1
            print(f"[{i}/{len(audio_ids)}] ok {result['anon_id']}", flush=True)

    elapsed = time.time() - t0
    report = f"""# Level-normalization processing report

- Recordings expected: {len(audio_ids)}
- Successfully processed: {n_ok}
- Failed: {len(failures)}
- Wall time: {elapsed:.1f}s ({n_workers} worker processes)

## Normalization settings

- Method: RMS-based, single linear gain per recording (see preprocessing/normalize.py)
- Target RMS: {TARGET_RMS} (fixed, label-independent)
- Peak safety ceiling: {PEAK_SAFETY} (gain capped, never hard-clipped)
- No resampling, no channel change, no denoising, no compression/limiting/EQ
- Channel 0 only; each recording normalized using only its own samples

## Feature settings (identical to preprocessing/features.py)

- Frame length 1024 (128ms), hop length 256 (32ms)
- F0: pYIN, fmin=50Hz, fmax=500Hz, unvoiced=NaN
- MFCC: 13 coefficients, mean+std summary

## Failures

{chr(10).join(f"- {r['anon_id']}: {r['reason']}" for r in failures) if failures else "(none)"}
"""
    with open(NORMALIZED_REPORT_MD, "w") as f:
        f.write(report)
    print(report)
    return {"ok": n_ok, "failed": len(failures), "elapsed_s": elapsed}


if __name__ == "__main__":
    run()
