"""
Live-pipeline A/B comparison -- exercises the REAL production path
(RollingDetector -> DecisionEngine -> EvidenceSession -> SpeakerGuard,
app/live_infer.py / app/decision.py, unmodified) for both CNN
checkpoints, with the silence gate ACTIVE (unlike the pure-CNN
diagnostics in modeling/diagnose_cnn_silence.py, which deliberately
bypassed it). This answers a different question: given the live system
AS IT ACTUALLY RUNS TODAY, how does each checkpoint behave?

Does not retrain, modify weights, thresholds, or app/decision.py. Reads
outputs/models/cnn_baseline.pt and cnn_speech_replaced.pt read-only.

Run with: python -m modeling.compare_live_pipeline
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from app.audio_utils import HOP_SECONDS
from app.live_infer import RollingDetector
from app.model import load_model

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "files" / "audio"
MODELS_DIR = REPO_ROOT / "outputs" / "models"
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"
DUMP_PATH = REPORTS_DIR / "_live_pipeline_comparison_dump.json"

KNOWN_HUMAN = ["call_0e1e2f29bfdc", "call_10aa0d1d33f0"]
KNOWN_SYNTH = ["call_0294f969f98b", "call_0847d7417bb1"]

dump = {}


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def run_recording_through_pipeline(model, anon_id):
    """Streams a real recording through the EXACT production RollingDetector,
    exactly as app/live_infer.py's run_from_file does, and returns the full
    per-window timeline."""
    y, sr = sf.read(str(AUDIO_DIR / f"{anon_id}.wav"), always_2d=True)
    y = y[:, 0].astype(np.float64)
    chunk_len = int(HOP_SECONDS * sr)
    detector = RollingDetector(model)
    timeline = []
    for start in range(0, len(y), chunk_len):
        chunk = y[start:start + chunk_len]
        if len(chunk) == 0:
            break
        detector.push(chunk, sr)
        if detector.ready():
            raw_rms = float(np.sqrt(np.mean(np.square(detector.buffer))))
            result = detector.step()
            state = ("SILENCE" if result["silence"] else
                     "SPEAKER_CHANGE" if result["speaker_change"] else
                     "STABILIZING" if result["stabilizing"] else "SPEECH")
            timeline.append({
                "t_start": start / sr, "rms": raw_rms, "signal_state": state,
                "raw": result["raw"], "smoothed": result["smoothed"],
                "decision_label": result["label"],
            })
    return timeline


def summarize_timeline(timeline: list) -> dict:
    speech_rows = [r for r in timeline if r["signal_state"] == "SPEECH"]
    raws = [r["raw"] for r in speech_rows if r["raw"] is not None]
    final_state = timeline[-1]["decision_label"] if timeline else None

    # "False AI events": count of transitions INTO "AI likely" from a
    # non-"AI likely" decision label (each confirmed flip, not every window)
    ai_confirmations, human_confirmations, flips = 0, 0, 0
    prev_label = None
    for r in timeline:
        lbl = r["decision_label"]
        if prev_label is not None and lbl != prev_label:
            flips += 1
            if lbl == "AI likely":
                ai_confirmations += 1
            if lbl == "Human likely":
                human_confirmations += 1
        prev_label = lbl

    return {
        "final_state": final_state,
        "mean_score": float(np.mean(raws)) if raws else None,
        "max_score": float(np.max(raws)) if raws else None,
        "min_score": float(np.min(raws)) if raws else None,
        "n_speech_windows": len(speech_rows),
        "n_silence_windows": sum(1 for r in timeline if r["signal_state"] == "SILENCE"),
        "state_flips": flips,
        "ai_confirmations": ai_confirmations,
        "human_confirmations": human_confirmations,
    }


def run():
    section("0. Load both checkpoints (read-only)")
    baseline = load_model(MODELS_DIR / "cnn_baseline.pt")
    speech_replaced = load_model(MODELS_DIR / "cnn_speech_replaced.pt")
    models = {"baseline": baseline, "speech_replaced": speech_replaced}
    print("Loaded cnn_baseline.pt and cnn_speech_replaced.pt.")

    section("1. Full live-pipeline timelines -- known human + synthetic recordings")
    all_timelines = {}
    all_summaries = {}
    for anon_id in KNOWN_HUMAN + KNOWN_SYNTH:
        for name, model in models.items():
            tl = run_recording_through_pipeline(model, anon_id)
            all_timelines[f"{anon_id}__{name}"] = tl
            s = summarize_timeline(tl)
            all_summaries[f"{anon_id}__{name}"] = s
            print(f"  {anon_id} [{name}]: final={s['final_state']} mean={s['mean_score']} "
                  f"max={s['max_score']} flips={s['state_flips']} "
                  f"ai_conf={s['ai_confirmations']} human_conf={s['human_confirmations']} "
                  f"(speech={s['n_speech_windows']} silence={s['n_silence_windows']})")

    section("2. Silence-persistence check: does evidence survive silence runs? (naturally occurring in the data)")
    persistence_examples = {}
    for anon_id in [KNOWN_SYNTH[0], KNOWN_HUMAN[0]]:
        for name, model in models.items():
            tl = all_timelines[f"{anon_id}__{name}"]
            # find a run of >=2 consecutive SILENCE rows with a confirmed
            # (AI likely / Human likely) decision immediately before AND after
            found = None
            i = 0
            while i < len(tl):
                if tl[i]["signal_state"] == "SILENCE":
                    j = i
                    while j < len(tl) and tl[j]["signal_state"] == "SILENCE":
                        j += 1
                    run_len = j - i
                    if run_len >= 2 and i > 0 and j < len(tl):
                        before = tl[i - 1]["decision_label"]
                        after = tl[j]["decision_label"]
                        if before in ("AI likely", "Human likely"):
                            found = {"run_start": i, "run_len": run_len, "before": before, "after": after,
                                     "rows": tl[max(0, i - 1):min(len(tl), j + 1)]}
                            break
                    i = j
                else:
                    i += 1
            if found:
                persistence_examples[f"{anon_id}__{name}"] = found
                print(f"\n  {anon_id} [{name}]: {found['run_len']}-window silence run, "
                      f"before='{found['before']}' after='{found['after']}' "
                      f"(persisted: {found['before'] == found['after']})")
                for r in found["rows"]:
                    print(f"    t={r['t_start']:6.1f}s rms={r['rms']:.4f} state={r['signal_state']:8s} "
                          f"raw={r['raw']} decision={r['decision_label']}")
            else:
                print(f"\n  {anon_id} [{name}]: no qualifying silence run found (recording may have too few pauses)")
    dump["persistence_examples"] = persistence_examples

    section("3. Quiet-human regression check: call_0e1e2f29bfdc through the FULL live pipeline")
    for name, model in models.items():
        tl = all_timelines[f"call_0e1e2f29bfdc__{name}"]
        # the previously-identified problematic raw indices (17,34,50,84 at 1.5s hop) -> t=25.5,51.0,75.0,126.0
        target_times = [25.5, 51.0, 75.0, 126.0]
        print(f"\n  [{name}] at the previously-flagged timestamps:")
        for t in target_times:
            row = min(tl, key=lambda r: abs(r["t_start"] - t)) if tl else None
            if row:
                print(f"    t={row['t_start']:6.1f}s rms={row['rms']:.5f} state={row['signal_state']:8s} "
                      f"raw={row['raw']} decision={row['decision_label']}")

    dump["timelines"] = all_timelines
    dump["summaries"] = all_summaries
    with open(DUMP_PATH, "w") as f:
        json.dump(dump, f, indent=2, default=str)
    print(f"\nRaw dump written to {DUMP_PATH}")
    return dump


if __name__ == "__main__":
    run()
