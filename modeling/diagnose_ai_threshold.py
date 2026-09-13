"""
DIAGNOSTIC-ONLY SCRIPT -- does not modify production code, weights, or
thresholds. Reads outputs/models/cnn_live_detector.pt (production, =
speech-replaced) and outputs/models/cnn_baseline.pt read-only.

Investigates a reported AI/synthetic under-detection issue after the
speech-replaced promotion: is it a CNN-score problem (scores genuinely
too low) or a decision-threshold problem (scores are reasonable but the
existing AI_THRESHOLD=0.65 rejects them)?

Threshold candidates are tested by re-implementing the EXACT SAME
smoothing/hysteresis math as app/decision.py:DecisionEngine, parameterized
by threshold -- app/decision.py itself is never imported for mutation and
never edited by this script. This lets candidate thresholds be evaluated
against the SAME real per-window score sequences without touching
production code.

Run with: python -m modeling.diagnose_ai_threshold
"""
import json
from collections import deque
from pathlib import Path

import numpy as np
import soundfile as sf

from app.audio_utils import HOP_SECONDS
from app.live_infer import RollingDetector
from app.model import load_model
from app.decision import (
    AI_THRESHOLD as PROD_AI_THRESHOLD, HUMAN_THRESHOLD as PROD_HUMAN_THRESHOLD,
    SMOOTHING_WINDOW as PROD_SMOOTHING_WINDOW, CONSECUTIVE_REQUIRED as PROD_CONSECUTIVE_REQUIRED,
    SILENCE_RMS_THRESHOLD as PROD_SILENCE_RMS_THRESHOLD,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "files" / "audio"
MODELS_DIR = REPO_ROOT / "outputs" / "models"
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"
DUMP_PATH = REPORTS_DIR / "_ai_threshold_diagnostic_dump.json"

HUMAN_RECORDINGS = ["call_0e1e2f29bfdc", "call_10aa0d1d33f0"]
SYNTH_RECORDINGS = ["call_0294f969f98b", "call_0847d7417bb1"]
CANDIDATE_AI_THRESHOLDS = [0.65, 0.60, 0.55, 0.50]

dump = {}


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def collect_scores(model, anon_id):
    """Streams a real recording through the ACTUAL production
    RollingDetector (silence gate, speaker guard, everything) and returns
    the raw per-window (is_silence, raw_score) sequence plus the real
    production decision timeline (using the real, unmodified DecisionEngine
    inside RollingDetector -- this IS app/decision.py running as-is)."""
    y, sr = sf.read(str(AUDIO_DIR / f"{anon_id}.wav"), always_2d=True)
    y = y[:, 0].astype(np.float64)
    chunk_len = int(HOP_SECONDS * sr)
    detector = RollingDetector(model)
    sequence = []  # (is_silence, raw_score_or_None)
    production_timeline = []
    for start in range(0, len(y), chunk_len):
        chunk = y[start:start + chunk_len]
        if len(chunk) == 0:
            break
        detector.push(chunk, sr)
        if detector.ready():
            result = detector.step()
            sequence.append((result["silence"], result["raw"]))
            production_timeline.append({"t": start / sr, "silence": result["silence"],
                                         "raw": result["raw"], "decision": result["label"]})
    return sequence, production_timeline


def score_stats(raws: list) -> dict:
    arr = np.array([r for r in raws if r is not None])
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)), "mean": float(np.mean(arr)), "median": float(np.median(arr)),
        "min": float(np.min(arr)), "max": float(np.max(arr)),
        "n_ge_035": int(np.sum(arr >= 0.35)), "n_ge_045": int(np.sum(arr >= 0.45)),
        "n_ge_050": int(np.sum(arr >= 0.50)), "n_ge_055": int(np.sum(arr >= 0.55)),
        "n_ge_060": int(np.sum(arr >= 0.60)), "n_ge_065": int(np.sum(arr >= 0.65)),
    }


def simulate_decision(sequence, ai_threshold, human_threshold=PROD_HUMAN_THRESHOLD,
                       smoothing_window=PROD_SMOOTHING_WINDOW, consecutive_required=PROD_CONSECUTIVE_REQUIRED):
    """Reimplements app/decision.py:DecisionEngine.update() EXACTLY
    (silence-as-cooldown, same smoothing/hysteresis math), parameterized
    by ai_threshold only, so candidate thresholds can be tested without
    touching the real module. Verified to match production output
    bit-for-bit at the production threshold (see verification step)."""
    history = deque(maxlen=smoothing_window)
    label = "Uncertain"
    candidate_label = None
    candidate_count = 0
    timeline = []
    ai_confirmations = 0
    human_confirmations = 0
    prev_label = None
    for is_silence, raw in sequence:
        if is_silence:
            timeline.append({"decision": "Silence"})
            prev_label = "Silence"
            continue
        history.append(raw)
        smoothed = sum(history) / len(history)
        if smoothed >= ai_threshold:
            candidate = "AI likely"
        elif smoothed <= human_threshold:
            candidate = "Human likely"
        else:
            candidate = "Uncertain"
        if candidate == candidate_label:
            candidate_count += 1
        else:
            candidate_label = candidate
            candidate_count = 1
        if candidate_count >= consecutive_required:
            label = candidate
        if label != prev_label:
            if label == "AI likely":
                ai_confirmations += 1
            elif label == "Human likely":
                human_confirmations += 1
        timeline.append({"decision": label})
        prev_label = label
    final = timeline[-1]["decision"] if timeline else None
    return {"final": final, "ai_confirmations": ai_confirmations,
            "human_confirmations": human_confirmations, "timeline": timeline}


def run():
    section("0. Load both checkpoints (read-only) + confirm decision.py values")
    baseline = load_model(MODELS_DIR / "cnn_baseline.pt")
    production = load_model(MODELS_DIR / "cnn_live_detector.pt")  # = speech-replaced weights
    print(f"AI_THRESHOLD={PROD_AI_THRESHOLD} HUMAN_THRESHOLD={PROD_HUMAN_THRESHOLD} "
          f"SMOOTHING_WINDOW={PROD_SMOOTHING_WINDOW} CONSECUTIVE_REQUIRED={PROD_CONSECUTIVE_REQUIRED} "
          f"SILENCE_RMS_THRESHOLD={PROD_SILENCE_RMS_THRESHOLD}")

    # --------------------------------------------------------------
    section("1. Score distributions -- PRODUCTION model (speech-replaced)")
    # --------------------------------------------------------------
    prod_data = {}
    for anon_id in HUMAN_RECORDINGS + SYNTH_RECORDINGS:
        seq, timeline = collect_scores(production, anon_id)
        stats = score_stats([r for _, r in seq])
        final_decision = timeline[-1]["decision"] if timeline else None
        prod_data[anon_id] = {"sequence": seq, "stats": stats, "final_decision": final_decision}
        print(f"  {anon_id}: n={stats.get('n')} mean={stats.get('mean')} median={stats.get('median')} "
              f"min={stats.get('min')} max={stats.get('max')} "
              f">=.35:{stats.get('n_ge_035')} >=.45:{stats.get('n_ge_045')} >=.50:{stats.get('n_ge_050')} "
              f">=.55:{stats.get('n_ge_055')} >=.60:{stats.get('n_ge_060')} >=.65:{stats.get('n_ge_065')} "
              f"| final_decision={final_decision}")

    # --------------------------------------------------------------
    section("2. Score distributions -- BASELINE model (comparison)")
    # --------------------------------------------------------------
    base_data = {}
    for anon_id in HUMAN_RECORDINGS + SYNTH_RECORDINGS:
        seq, timeline = collect_scores(baseline, anon_id)
        stats = score_stats([r for _, r in seq])
        final_decision = timeline[-1]["decision"] if timeline else None
        base_data[anon_id] = {"stats": stats, "final_decision": final_decision}
        print(f"  {anon_id}: mean={stats.get('mean')} median={stats.get('median')} max={stats.get('max')} "
              f"| final_decision={final_decision}")

    print("\nComparison table:")
    print(f"{'Recording':22s} {'Label':10s} {'Base mean':>10s} {'Repl mean':>10s} {'Base max':>9s} {'Repl max':>9s} {'Decision (repl)':>16s}")
    for anon_id in HUMAN_RECORDINGS + SYNTH_RECORDINGS:
        label = "human" if anon_id in HUMAN_RECORDINGS else "synthetic"
        b = base_data[anon_id]["stats"]
        r = prod_data[anon_id]["stats"]
        print(f"{anon_id:22s} {label:10s} {b.get('mean', 0):10.3f} {r.get('mean', 0):10.3f} "
              f"{b.get('max', 0):9.3f} {r.get('max', 0):9.3f} {prod_data[anon_id]['final_decision']:>16s}")

    # --------------------------------------------------------------
    section("3. Verify the simulated decision math matches the REAL production DecisionEngine exactly")
    # --------------------------------------------------------------
    all_match = True
    for anon_id in HUMAN_RECORDINGS + SYNTH_RECORDINGS:
        seq, real_timeline = collect_scores(production, anon_id)
        sim = simulate_decision(seq, PROD_AI_THRESHOLD)
        real_decisions = [t["decision"] for t in real_timeline]
        sim_decisions = [t["decision"] for t in sim["timeline"]]
        match = real_decisions == sim_decisions
        all_match = all_match and match
        print(f"  {anon_id}: simulator matches real DecisionEngine at threshold={PROD_AI_THRESHOLD}: {match}")
    print(f"\nSimulator verified {'CORRECT' if all_match else 'MISMATCHED -- DO NOT TRUST THRESHOLD SIMULATION'}")
    dump["simulator_verified"] = all_match

    # --------------------------------------------------------------
    section("4. Does synthetic speech consistently score below 0.65? (gate for whether to test thresholds at all)")
    # --------------------------------------------------------------
    synth_below_65 = all(prod_data[a]["stats"].get("mean", 1.0) < 0.65 or prod_data[a]["final_decision"] != "AI likely"
                          for a in SYNTH_RECORDINGS)
    for a in SYNTH_RECORDINGS:
        s = prod_data[a]["stats"]
        print(f"  {a}: mean={s.get('mean'):.3f} n>=.65={s.get('n_ge_065')}/{s.get('n')} final={prod_data[a]['final_decision']}")
    print(f"\nProceeding to threshold testing: {synth_below_65}")

    threshold_results = {}
    if synth_below_65:
        section("5. Threshold candidate testing (simulation only, production code untouched)")
        for thr in CANDIDATE_AI_THRESHOLDS:
            print(f"\n--- AI_THRESHOLD = {thr} ---")
            threshold_results[thr] = {}
            for anon_id in HUMAN_RECORDINGS + SYNTH_RECORDINGS:
                seq = prod_data[anon_id]["sequence"]
                sim = simulate_decision(seq, thr)
                threshold_results[thr][anon_id] = {
                    "final": sim["final"], "ai_confirmations": sim["ai_confirmations"],
                    "human_confirmations": sim["human_confirmations"],
                }
                label = "human" if anon_id in HUMAN_RECORDINGS else "synthetic"
                print(f"    {anon_id:22s} ({label:9s}) final={sim['final']:>14s} "
                      f"ai_conf={sim['ai_confirmations']} human_conf={sim['human_confirmations']}")
    else:
        print("\nSkipping threshold testing -- synthetic speech does NOT consistently score below 0.65; "
              "under-detection is not primarily a threshold problem.")

    dump["production_stats"] = {k: v["stats"] for k, v in prod_data.items()}
    dump["production_final_decisions"] = {k: v["final_decision"] for k, v in prod_data.items()}
    dump["baseline_stats"] = {k: v["stats"] for k, v in base_data.items()}
    dump["baseline_final_decisions"] = {k: v["final_decision"] for k, v in base_data.items()}
    dump["threshold_results"] = {str(k): v for k, v in threshold_results.items()}
    with open(DUMP_PATH, "w") as f:
        json.dump(dump, f, indent=2, default=str)
    print(f"\nRaw dump written to {DUMP_PATH}")
    return dump


if __name__ == "__main__":
    run()
