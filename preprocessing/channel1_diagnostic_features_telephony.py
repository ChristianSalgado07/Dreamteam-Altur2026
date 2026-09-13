"""
Channel-1 (confirmed AI agent) acoustic features on telephony-band-
limited audio — both variants (band-limited only, and band-limited +
RMS-normalized) — the Channel-1 counterpart of
preprocessing/telephony_pipeline.py, used only for the telephony
experiment's Channel-1 diagnostic. Same fast, non-pYIN feature scope as
the other channel1_diagnostic_features_*.py scripts (RMS, spectral,
ZCR, MFCC).

Writes:
  outputs/features/channel1_diagnostic_features_telephony.csv
  outputs/features/channel1_diagnostic_features_telephony_normalized.csv

Run with: python -m preprocessing.channel1_diagnostic_features_telephony
"""
import csv
import multiprocessing as mp
import time

from preprocessing.audio_io import AUDIO_DIR, load_stereo, basic_signal_stats
from preprocessing.telephony_filter import apply_telephony_band
from preprocessing.normalize import normalize_rms
from preprocessing.features import extract_rms, extract_spectral, extract_mfcc, mfcc_summary, nan_summary
from preprocessing.pipeline import FEATURES_DIR, REPORTS_DIR

CHANNEL1_TELEPHONY_CSV = FEATURES_DIR / "channel1_diagnostic_features_telephony.csv"
CHANNEL1_TELEPHONY_NORMALIZED_CSV = FEATURES_DIR / "channel1_diagnostic_features_telephony_normalized.csv"
REPORT_MD = REPORTS_DIR / "channel1_diagnostic_telephony_report.md"


def _features_row(y, sr, anon_id):
    stats = basic_signal_stats(y, sr)
    rms = extract_rms(y)
    centroid, bandwidth, rolloff, zcr = extract_spectral(y, sr)
    mfcc = extract_mfcc(y, sr)
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
    return row


def process_one(anon_id: str):
    try:
        _, ch1, sr = load_stereo(anon_id)
        ch1_band, _ = apply_telephony_band(ch1, sr)
        row_a = _features_row(ch1_band, sr, anon_id)

        ch1_band_norm, _ = normalize_rms(ch1_band)
        row_b = _features_row(ch1_band_norm, sr, anon_id)

        return {"status": "ok", "anon_id": anon_id, "row_a": row_a, "row_b": row_b}
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
    writer_a = writer_b = None

    with open(CHANNEL1_TELEPHONY_CSV, "w", newline="") as f_a, \
         open(CHANNEL1_TELEPHONY_NORMALIZED_CSV, "w", newline="") as f_b, \
         mp.Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one, audio_ids), start=1):
            if result["status"] != "ok":
                failures.append(result)
                print(f"[{i}/{len(audio_ids)}] FAILED {result['anon_id']}: {result['reason']}", flush=True)
                continue
            if writer_a is None:
                writer_a = csv.DictWriter(f_a, fieldnames=list(result["row_a"].keys()))
                writer_a.writeheader()
                writer_b = csv.DictWriter(f_b, fieldnames=list(result["row_b"].keys()))
                writer_b.writeheader()
            writer_a.writerow(result["row_a"])
            writer_b.writerow(result["row_b"])
            f_a.flush()
            f_b.flush()
            n_ok += 1
            print(f"[{i}/{len(audio_ids)}] ok {result['anon_id']}", flush=True)

    elapsed = time.time() - t0
    report = f"""# Channel 1 diagnostic (telephony-band) feature extraction report

- Recordings expected: {len(audio_ids)}
- Successfully processed: {n_ok}
- Failed: {len(failures)}
- Wall time: {elapsed:.1f}s ({n_workers} worker processes)

## Failures

{chr(10).join(f"- {r['anon_id']}: {r['reason']}" for r in failures) if failures else "(none)"}
"""
    with open(REPORT_MD, "w") as f:
        f.write(report)
    print(report)
    return {"ok": n_ok, "failed": len(failures), "elapsed_s": elapsed}


if __name__ == "__main__":
    run()
