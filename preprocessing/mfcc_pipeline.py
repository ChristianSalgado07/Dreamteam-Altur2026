"""
MFCC recording-level summary features — a small, additive companion to
preprocessing/pipeline.py (which already produced recording_level.csv /
timeseries.csv without MFCCs). Kept as its own script rather than
folded into pipeline.py so the already-completed acoustic run doesn't
need to be regenerated; reuses the same audio loading / denoising
utilities and frame settings (preprocessing/audio_io.py,
preprocessing/denoise.py, preprocessing/features.py FRAME_LENGTH/HOP_LENGTH).

Writes outputs/features/mfcc_recording_level.csv — one row per
(anon_id, version) matching recording_level.csv's shape, so the two can
be joined directly on (anon_id, version).

Run with: python -m preprocessing.mfcc_pipeline
"""
import csv
import multiprocessing as mp
import time

from preprocessing.audio_io import AUDIO_DIR, load_channel0, basic_signal_stats
from preprocessing.denoise import denoise_channel
from preprocessing.features import extract_mfcc, mfcc_summary
from preprocessing.pipeline import FEATURES_DIR, REPORTS_DIR

MFCC_RECORDING_LEVEL_CSV = FEATURES_DIR / "mfcc_recording_level.csv"
MFCC_REPORT_MD = REPORTS_DIR / "mfcc_processing_report.md"


def process_one(anon_id: str):
    try:
        y, sr = load_channel0(anon_id)
        y_dn, _ = denoise_channel(y, sr)
        rows = []
        for version, signal in (("original", y), ("denoised", y_dn)):
            mfcc = extract_mfcc(signal, sr)
            row = {"anon_id": anon_id, "version": version}
            row.update(mfcc_summary(mfcc))
            rows.append(row)
        return {"status": "ok", "anon_id": anon_id, "rows": rows}
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

    with open(MFCC_RECORDING_LEVEL_CSV, "w", newline="") as f, \
         mp.Pool(processes=n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one, audio_ids), start=1):
            if result["status"] != "ok":
                failures.append(result)
                print(f"[{i}/{len(audio_ids)}] FAILED {result['anon_id']}: {result['reason']}", flush=True)
                continue
            for row in result["rows"]:
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
            f.flush()
            n_ok += 1
            print(f"[{i}/{len(audio_ids)}] ok {result['anon_id']}", flush=True)

    elapsed = time.time() - t0
    report = f"""# MFCC extraction report

- Recordings expected: {len(audio_ids)}
- Successfully processed: {n_ok}
- Failed: {len(failures)}
- Wall time: {elapsed:.1f}s ({n_workers} worker processes)

## Settings

- N_MFCC = 13, frame_length=1024 (128ms), hop_length=256 (32ms) — same
  frame grid as preprocessing/features.py's other acoustic features
- Computed for both "original" and "denoised" Channel-0 signals
- Recording-level summary only (per-coefficient mean + std); no MFCC
  time-series was added to outputs/features/timeseries.csv

## Failures

{chr(10).join(f"- {r['anon_id']}: {r['reason']}" for r in failures) if failures else "(none)"}
"""
    with open(MFCC_REPORT_MD, "w") as f:
        f.write(report)
    print(report)
    return {"ok": n_ok, "failed": len(failures), "elapsed_s": elapsed}


if __name__ == "__main__":
    run()
