"""
Foundational acoustic preprocessing / feature-extraction pipeline.

For every recording in files/audio:
  1. load the stereo wav, isolate Channel 0 (never touches the raw file)
  2. compute basic signal-inspection stats on the raw Channel 0 signal
  3. build a noise-reduced copy of Channel 0 (see preprocessing/denoise.py)
  4. extract F0 / RMS / spectral features from BOTH the original and the
     noise-reduced signal (see preprocessing/features.py)
  5. attach the human/synthetic label from files/csv/manifest.csv
  6. write everything to outputs/ (never back into files/)

Run with:  python -m preprocessing.pipeline
Multiprocessing is used only to parallelize independent per-file work
(no cross-file state, no model training) because pYIN pitch tracking
is the slow step and there are 353 recordings x 2 signal versions.
"""
import csv
import multiprocessing as mp
import time
from pathlib import Path

from preprocessing.audio_io import (
    AUDIO_DIR,
    JSON_DIR,
    MANIFEST_CSV,
    basic_signal_stats,
    load_channel0,
)
from preprocessing.denoise import denoise_channel
from preprocessing.features import extract_all

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
FEATURES_DIR = OUTPUT_DIR / "features"
REPORTS_DIR = OUTPUT_DIR / "reports"

RECORDING_LEVEL_CSV = FEATURES_DIR / "recording_level.csv"
TIMESERIES_CSV = FEATURES_DIR / "timeseries.csv"
PROCESSING_REPORT_CSV = REPORTS_DIR / "processing_report.csv"
PROCESSING_REPORT_MD = REPORTS_DIR / "processing_report.md"


def load_manifest() -> dict:
    with open(MANIFEST_CSV, newline="") as f:
        return {row["anon_id"]: row for row in csv.DictReader(f)}


def has_turn_data(anon_id: str) -> bool:
    path = JSON_DIR / f"{anon_id}.json"
    return path.exists() and path.stat().st_size > 0


def _feature_row(anon_id: str, version: str, stats: dict, summary: dict, manifest_row: dict) -> dict:
    row = {
        "anon_id": anon_id,
        "label": manifest_row["label"],
        "split": manifest_row.get("split", ""),
        "channel": 0,
        "version": version,  # "original" or "denoised"
        "has_turn_data": has_turn_data(anon_id),
        "duration_s": stats["duration_s"],
        "sample_rate": stats["sample_rate"],
        "n_samples": stats["n_samples"],
        "raw_peak_amplitude": stats["peak_amplitude"],
        "raw_rms": stats["rms"],
        "raw_silence_proportion": stats["silence_proportion"],
    }
    row.update(summary)
    return row


def process_one(anon_id: str):
    """
    Worker function (runs in a subprocess). Returns a dict describing
    what happened, never raises — failures are captured and reported
    rather than silently skipped.
    """
    try:
        y, sr = load_channel0(anon_id)
        stats = basic_signal_stats(y, sr)

        y_dn, denoise_params = denoise_channel(y, sr)

        ts_orig, summary_orig = extract_all(y, sr)
        ts_dn, summary_dn = extract_all(y_dn, sr)

        return {
            "status": "ok",
            "anon_id": anon_id,
            "stats": stats,
            "denoise_params": denoise_params,
            "timeseries": {"original": ts_orig, "denoised": ts_dn},
            "summary": {"original": summary_orig, "denoised": summary_dn},
        }
    except Exception as e:
        return {
            "status": "failed",
            "anon_id": anon_id,
            "error_type": type(e).__name__,
            "error_message": str(e),
        }


def run(n_workers: int = 6, limit: int = None):
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    audio_ids = sorted(p.stem for p in AUDIO_DIR.glob("*.wav"))
    if limit:
        audio_ids = audio_ids[:limit]

    expected_n = len(audio_ids)
    successes = []
    failures = []
    turn_data_present = 0
    turn_data_missing = 0

    t_start = time.time()

    with open(RECORDING_LEVEL_CSV, "w", newline="") as f_rec, \
         open(TIMESERIES_CSV, "w", newline="") as f_ts:

        rec_writer = None
        ts_writer = None

        with mp.Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(process_one, audio_ids), start=1):
                anon_id = result["anon_id"]

                if result["status"] != "ok":
                    failures.append({
                        "anon_id": anon_id,
                        "reason": f"{result['error_type']}: {result['error_message']}",
                    })
                    print(f"[{i}/{expected_n}] FAILED  {anon_id}: {result['error_message']}")
                    continue

                manifest_row = manifest.get(anon_id)
                if manifest_row is None:
                    failures.append({
                        "anon_id": anon_id,
                        "reason": "no matching row in manifest.csv",
                    })
                    print(f"[{i}/{expected_n}] FAILED  {anon_id}: not in manifest.csv")
                    continue

                td = has_turn_data(anon_id)
                turn_data_present += int(td)
                turn_data_missing += int(not td)

                for version in ("original", "denoised"):
                    row = _feature_row(
                        anon_id, version, result["stats"],
                        result["summary"][version], manifest_row,
                    )
                    if rec_writer is None:
                        rec_writer = csv.DictWriter(f_rec, fieldnames=list(row.keys()))
                        rec_writer.writeheader()
                    rec_writer.writerow(row)

                    ts = result["timeseries"][version]
                    n = len(ts["time"])
                    for j in range(n):
                        ts_row = {
                            "anon_id": anon_id,
                            "version": version,
                            "time": ts["time"][j],
                            "f0": ts["f0"][j],
                            "voiced_flag": bool(ts["voiced_flag"][j]),
                            "rms": ts["rms"][j],
                            "spectral_centroid": ts["spectral_centroid"][j],
                            "spectral_bandwidth": ts["spectral_bandwidth"][j],
                            "spectral_rolloff": ts["spectral_rolloff"][j],
                            "zcr": ts["zcr"][j],
                        }
                        if ts_writer is None:
                            ts_writer = csv.DictWriter(f_ts, fieldnames=list(ts_row.keys()))
                            ts_writer.writeheader()
                        ts_writer.writerow(ts_row)

                successes.append(anon_id)
                print(f"[{i}/{expected_n}] ok      {anon_id}")

    elapsed = time.time() - t_start

    # ---- processing report ----
    with open(PROCESSING_REPORT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["anon_id", "status", "reason"])
        for aid in successes:
            w.writerow([aid, "ok", ""])
        for entry in failures:
            w.writerow([entry["anon_id"], "failed", entry["reason"]])

    report_md = f"""# Acoustic preprocessing — processing report

- Expected recordings: {expected_n}
- Successfully processed: {len(successes)}
- Failed: {len(failures)}
- Recordings with valid turn metadata (has_turn_data=True): {turn_data_present}
- Recordings with empty turn metadata (has_turn_data=False): {turn_data_missing}
- Total wall time: {elapsed:.1f}s ({n_workers} worker processes)

## Preprocessing settings

- Sample rate: 8000 Hz (native, unchanged)
- Channel used: 0 (target speaker) only
- Frame length: 1024 samples (128 ms), hop length: 256 samples (32 ms)
- F0 method: librosa pYIN, fmin=50 Hz, fmax=500 Hz, unvoiced frames = NaN
- Noise reduction: Butterworth high-pass filter, cutoff=80 Hz, order=4, zero-phase (filtfilt)
  applied to a SEPARATE copy of Channel 0 ("denoised"); the "original" copy is
  always also processed and kept.

## Failures

{chr(10).join(f"- {e['anon_id']}: {e['reason']}" for e in failures) if failures else "(none)"}
"""
    with open(PROCESSING_REPORT_MD, "w") as f:
        f.write(report_md)

    print(report_md)
    return {
        "expected": expected_n,
        "ok": len(successes),
        "failed": len(failures),
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    run()
