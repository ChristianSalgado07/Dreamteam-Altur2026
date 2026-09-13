"""
Demo-validation sanity check, NOT a metrics-tuning exercise: runs the
live detector's exact code path (RollingDetector.step(), including the
fixed app/decision.py thresholds) over a few existing labeled
recordings and reports what actually comes out. Thresholds are not
adjusted based on these results.

Run with: python -m app.test_known_examples
Writes: outputs/reports/known_examples_test.md
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
REPORT_MD = REPO_ROOT / "outputs" / "reports" / "known_examples_test.md"

N_PER_LABEL = 2  # same fixed selection rule as app/benchmark_latency.py


def pick_recordings() -> list:
    manifest = pd.read_csv(MANIFEST_CSV)
    val = manifest[manifest["split"] == "val"]
    rows = []
    for label in ("human", "synthetic"):
        rows += val[val["label"] == label].head(N_PER_LABEL).to_dict("records")
    return rows


def run():
    model_path = default_model_path()
    if not model_path.exists():
        print(f"Model not found at {model_path}. Run `python -m modeling.train_cnn` first.")
        return
    model = load_model(model_path)

    rows = pick_recordings()
    print(f"Testing on {len(rows)} known recordings (2 human, 2 synthetic, val split)\n")

    results = []
    for row in rows:
        anon_id, true_label = row["anon_id"], row["label"]
        y, sr = sf.read(str(AUDIO_DIR / f"{anon_id}.wav"), always_2d=True)
        y = y[:, 0]
        chunk_len = int(HOP_SECONDS * sr)
        detector = RollingDetector(model)
        raw_scores = []
        final_label = "Uncertain"
        for start in range(0, len(y), chunk_len):
            chunk = y[start:start + chunk_len]
            if len(chunk) == 0:
                break
            detector.push(chunk, sr)
            if detector.ready():
                result = detector.step()
                if result["error"] is None and not result["silence"]:
                    raw_scores.append(result["raw"])
                    final_label = result["label"]

        entry = {
            "anon_id": anon_id,
            "true_label": true_label,
            "n_windows": len(raw_scores),
            "mean_score": statistics.mean(raw_scores) if raw_scores else None,
            "median_score": statistics.median(raw_scores) if raw_scores else None,
            "final_smoothed_label": final_label,
        }
        results.append(entry)
        print(f"  {anon_id} (true={true_label}): n={entry['n_windows']} "
              f"mean={entry['mean_score']:.3f} median={entry['median_score']:.3f} "
              f"final_label={final_label}" if raw_scores else
              f"  {anon_id} (true={true_label}): no non-silent windows scored")

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Known-examples demo-validation test\n",
        "Sanity check only -- NOT a metrics-tuning exercise. Runs the live "
        "detector's exact code path (RollingDetector.step(), including the "
        "fixed app/decision.py thresholds: AI >= 0.90, Human <= 0.10, "
        "2 consecutive windows required to flip the displayed label) over "
        "2 human + 2 synthetic recordings from the existing val split. "
        "Thresholds were NOT adjusted based on these results.\n",
        f"\n```\n{json.dumps(results, indent=2)}\n```\n",
        "\nNote: this is 4 recordings, not a validation set -- see "
        "outputs/reports/cnn_training_report.md for the actual internal/"
        "external metrics (recording-level ROC-AUC 1.00 internal, 0.53 "
        "external).\n",
    ]
    with open(REPORT_MD, "w") as f:
        f.writelines(lines)
    print(f"\nSaved -> {REPORT_MD}")


if __name__ == "__main__":
    run()
