"""
Channel-1 (confirmed AI agent) acoustic feature extraction — for the
modeling-diagnostics Channel-1 comparison ONLY (modeling/diagnostics.py
section 11), not for training the Channel-0 classifier.

Deliberately skips F0/pYIN (the slow part of the acoustic pipeline,
~25 minutes for 353 recordings) since the diagnostic only needs the
non-pitch features that turned up as the strongest separators so far
(RMS, spectral centroid/bandwidth/rolloff, ZCR, MFCCs) — all fast,
non-pitch-tracking computations. If F0 comparison for Channel 1 is
needed later, extend this script rather than the completed Channel-0
pipeline.

Writes outputs/features/channel1_diagnostic_features.csv — one row per
anon_id, ORIGINAL signal only (denoising not needed for this
diagnostic). Never touches files/ or any existing Channel-0 output.

Run with: python -m preprocessing.channel1_diagnostic_features
"""
import csv
import multiprocessing as mp
import time

from preprocessing.audio_io import AUDIO_DIR, load_stereo, basic_signal_stats
from preprocessing.features import extract_rms, extract_spectral, extract_mfcc, mfcc_summary, nan_summary
from preprocessing.pipeline import FEATURES_DIR, REPORTS_DIR

CHANNEL1_CSV = FEATURES_DIR / "channel1_diagnostic_features.csv"
CHANNEL1_REPORT_MD = REPORTS_DIR / "channel1_diagnostic_report.md"


def process_one(anon_id: str):
    try:
        _, ch1, sr = load_stereo(anon_id)
        stats = basic_signal_stats(ch1, sr)

        rms = extract_rms(ch1)
        centroid, bandwidth, rolloff, zcr = extract_spectral(ch1, sr)
        mfcc = extract_mfcc(ch1, sr)

        row = {"anon_id": anon_id, "channel": 1}
        row["raw_peak_amplitude"] = stats["peak_amplitude"]
        row["raw_rms"] = stats["rms"]
        row["raw_silence_proportion"] = stats["silence_proportion"]
        row.update(nan_summary(rms, "rms"))
        row.update(nan_summary(centroid, "spectral_centroid"))
        row.update(nan_summary(bandwidth, "spectral_bandwidth"))
        row.update(nan_summary(rolloff, "spectral_rolloff"))
        row.update(nan_summary(zcr, "zcr"))
        row.update(mfcc_summary(mfcc))
        return {"status": "ok", "anon_id": anon_id, "row": row}
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
    writer = None

    with open(CHANNEL1_CSV, "w", newline="") as f, mp.Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one, audio_ids), start=1):
            if result["status"] != "ok":
                failures.append(result)
                print(f"[{i}/{len(audio_ids)}] FAILED {result['anon_id']}: {result['reason']}", flush=True)
                continue
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(result["row"].keys()))
                writer.writeheader()
            writer.writerow(result["row"])
            f.flush()
            n_ok += 1
            print(f"[{i}/{len(audio_ids)}] ok {result['anon_id']}", flush=True)

    elapsed = time.time() - t0
    report = f"""# Channel 1 diagnostic feature extraction report

- Recordings expected: {len(audio_ids)}
- Successfully processed: {n_ok}
- Failed: {len(failures)}
- Wall time: {elapsed:.1f}s ({n_workers} worker processes)

Scope: RMS, spectral centroid/bandwidth/rolloff, ZCR, MFCC 1-13 (mean+std)
on Channel 1, ORIGINAL signal only. F0/pYIN was intentionally skipped
(slow; not needed for the modeling-diagnostics comparison this supports).

## Failures

{chr(10).join(f"- {r['anon_id']}: {r['reason']}" for r in failures) if failures else "(none)"}
"""
    with open(CHANNEL1_REPORT_MD, "w") as f:
        f.write(report)
    print(report)
    return {"ok": n_ok, "failed": len(failures), "elapsed_s": elapsed}


if __name__ == "__main__":
    run()
