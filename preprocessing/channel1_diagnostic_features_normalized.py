"""
Channel-1 (confirmed AI agent) acoustic features on RMS-level-normalized
audio — the Channel-1 counterpart of preprocessing/normalized_pipeline.py,
used only for the normalization experiment's Channel-1 diagnostic
(step 10). Mirrors preprocessing/channel1_diagnostic_features.py (which
stays untouched, original-audio only) — same fast, non-pYIN feature
scope (RMS, spectral, ZCR, MFCC), same normalization as Channel 0
(preprocessing/normalize.py), applied independently to Channel 1's own
samples.

Writes outputs/features/channel1_diagnostic_features_normalized.csv.

Run with: python -m preprocessing.channel1_diagnostic_features_normalized
"""
import csv
import multiprocessing as mp
import time

from preprocessing.audio_io import AUDIO_DIR, load_stereo, basic_signal_stats
from preprocessing.normalize import normalize_rms
from preprocessing.features import extract_rms, extract_spectral, extract_mfcc, mfcc_summary, nan_summary
from preprocessing.pipeline import FEATURES_DIR, REPORTS_DIR

CHANNEL1_NORMALIZED_CSV = FEATURES_DIR / "channel1_diagnostic_features_normalized.csv"
CHANNEL1_NORMALIZED_REPORT_MD = REPORTS_DIR / "channel1_diagnostic_normalized_report.md"


def process_one(anon_id: str):
    try:
        _, ch1, sr = load_stereo(anon_id)
        ch1_norm, _ = normalize_rms(ch1)
        stats = basic_signal_stats(ch1_norm, sr)

        rms = extract_rms(ch1_norm)
        centroid, bandwidth, rolloff, zcr = extract_spectral(ch1_norm, sr)
        mfcc = extract_mfcc(ch1_norm, sr)

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

    with open(CHANNEL1_NORMALIZED_CSV, "w", newline="") as f, mp.Pool(processes=n_workers) as pool:
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
    report = f"""# Channel 1 diagnostic (RMS-normalized) feature extraction report

- Recordings expected: {len(audio_ids)}
- Successfully processed: {n_ok}
- Failed: {len(failures)}
- Wall time: {elapsed:.1f}s ({n_workers} worker processes)

Same scope as preprocessing/channel1_diagnostic_features.py (RMS,
spectral, ZCR, MFCC; F0/pYIN skipped), applied to RMS-normalized
Channel 1 audio (preprocessing/normalize.py, same settings as Channel 0).

## Failures

{chr(10).join(f"- {r['anon_id']}: {r['reason']}" for r in failures) if failures else "(none)"}
"""
    with open(CHANNEL1_NORMALIZED_REPORT_MD, "w") as f:
        f.write(report)
    print(report)
    return {"ok": n_ok, "failed": len(failures), "elapsed_s": elapsed}


if __name__ == "__main__":
    run()
