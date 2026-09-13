"""
Measures whether the live detector is actually fast enough for a
convincing live demo. Runs RollingDetector.step() (the SAME path used by
the microphone, --file, and FastAPI front ends) over a few existing
recordings and reports preprocessing / inference / total per-window
timing. Does NOT optimize anything -- measurement only, per the task
("do not optimize prematurely, first measure").

Run with: python -m app.benchmark_latency
Writes: outputs/reports/latency_benchmark.md
"""
import json
import statistics
from pathlib import Path

import pandas as pd
import soundfile as sf

from app.audio_utils import HOP_SECONDS
from app.live_infer import RollingDetector, default_model_path
from app.model import load_model

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_CSV = REPO_ROOT / "files" / "csv" / "manifest.csv"
AUDIO_DIR = REPO_ROOT / "files" / "audio"
REPORT_MD = REPO_ROOT / "outputs" / "reports" / "latency_benchmark.md"

# A small, fixed, non-cherry-picked sample: first val-split human and
# synthetic recording in manifest order (same selection rule as
# app/test_known_examples.py, so both scripts exercise the same clips).
N_PER_LABEL = 2


def pick_recordings() -> list:
    manifest = pd.read_csv(MANIFEST_CSV)
    val = manifest[manifest["split"] == "val"]
    ids = []
    for label in ("human", "synthetic"):
        ids += val[val["label"] == label]["anon_id"].head(N_PER_LABEL).tolist()
    return ids


def run():
    model_path = default_model_path()
    if not model_path.exists():
        print(f"Model not found at {model_path}. Run `python -m modeling.train_cnn` first.")
        return
    model = load_model(model_path)

    anon_ids = pick_recordings()
    print(f"Benchmarking on {len(anon_ids)} recordings: {anon_ids}\n")

    preprocess_ms, inference_ms, total_ms = [], [], []
    per_recording = []

    for anon_id in anon_ids:
        y, sr = sf.read(str(AUDIO_DIR / f"{anon_id}.wav"), always_2d=True)
        y = y[:, 0]
        chunk_len = int(HOP_SECONDS * sr)
        detector = RollingDetector(model)
        rec_total = []
        for start in range(0, len(y), chunk_len):
            chunk = y[start:start + chunk_len]
            if len(chunk) == 0:
                break
            detector.push(chunk, sr)
            if detector.ready():
                result = detector.step()
                if result["error"] is None and result["total_ms"] is not None:
                    total_ms.append(result["total_ms"])
                    rec_total.append(result["total_ms"])
                    if result["preprocess_ms"] is not None:
                        preprocess_ms.append(result["preprocess_ms"])
                        inference_ms.append(result["inference_ms"])
        per_recording.append({"anon_id": anon_id, "n_windows": len(rec_total),
                               "mean_total_ms": statistics.mean(rec_total) if rec_total else None})
        print(f"  {anon_id}: {len(rec_total)} scored windows")

    def summarize(values):
        if not values:
            return None
        values_sorted = sorted(values)
        p95_idx = int(0.95 * (len(values_sorted) - 1))
        return {
            "n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values), "p95": values_sorted[p95_idx],
        }

    summary = {
        "preprocess_ms": summarize(preprocess_ms),
        "inference_ms": summarize(inference_ms),
        "total_ms": summarize(total_ms),
        "hop_seconds": HOP_SECONDS,
        "hop_ms": HOP_SECONDS * 1000,
    }

    print("\n=== Latency summary (ms/window, excludes silent windows which skip the model) ===")
    print(json.dumps(summary, indent=2))

    mean_total = summary["total_ms"]["mean"] if summary["total_ms"] else None
    keeps_up = mean_total is not None and mean_total < summary["hop_ms"]
    headroom = (summary["hop_ms"] / mean_total) if mean_total else None
    print(f"\nHop interval: {summary['hop_ms']:.0f}ms. Mean total processing: "
          f"{mean_total:.1f}ms. {'Keeps up with real-time' if keeps_up else 'DOES NOT keep up with real-time'} "
          f"(headroom ~{headroom:.1f}x)." if mean_total else "")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Live-detector latency benchmark\n",
        f"Measured on {len(anon_ids)} existing val-split recordings "
        f"({anon_ids}), running RollingDetector.step() -- the identical "
        "code path used by the microphone, `--file`, and FastAPI front "
        "ends. CPU only (no GPU used anywhere in this project).\n",
        f"\nHop interval (window cadence): {summary['hop_ms']:.0f} ms.\n",
        "\n## Per-window timing (ms)\n",
        f"```\n{json.dumps(summary, indent=2)}\n```\n",
        f"\n**{'Keeps up with real-time' if keeps_up else 'Does NOT keep up with real-time'}** "
        f"-- mean total processing time is "
        f"{'well under' if keeps_up else 'over'} the {summary['hop_ms']:.0f}ms hop interval "
        f"(~{headroom:.1f}x headroom).\n" if mean_total else "\nNo non-silent windows were scored.\n",
        "\n## Per-recording\n",
        f"```\n{json.dumps(per_recording, indent=2)}\n```\n",
    ]
    with open(REPORT_MD, "w") as f:
        f.writelines(lines)
    print(f"\nSaved -> {REPORT_MD}")


if __name__ == "__main__":
    run()
