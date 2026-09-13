"""
Temporal / conversational analysis pipeline.

For every recording with valid turn metadata (258 of 353):
  1. load + validate the JSON turns (temporal/turns_io.py)
  2. compute target-speaker turn stats, other-speaker turn stats,
     turn-taking / transition stats, response-latency stats, candidate
     within-turn-pause stats (using the ORIGINAL Channel-0 RMS
     time-series already computed by the acoustic pipeline), and
     temporal-variability stats (temporal/turn_features.py)
  3. write a recording-level table, a turn-event table, and a
     within-turn-pause event table to outputs/features/
  4. write a validation + processing report to outputs/reports/

The 95 recordings with empty JSON are excluded from all turn-level
output (no fabricated turn boundaries) but are listed in the report.
Does not touch files/ or the existing acoustic outputs
(recording_level.csv / timeseries.csv are only READ, never rewritten).

Run with: python -m temporal.pipeline
"""
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.audio_io import MANIFEST_CSV
from preprocessing.pipeline import OUTPUT_DIR, FEATURES_DIR, REPORTS_DIR
from temporal.turns_io import load_and_validate_turns, validate_all
from temporal import turn_features as tf

RECORDING_LEVEL_ACOUSTIC_CSV = FEATURES_DIR / "recording_level.csv"
TIMESERIES_CSV = FEATURES_DIR / "timeseries.csv"

TEMPORAL_RECORDING_LEVEL_CSV = FEATURES_DIR / "temporal_recording_level.csv"
TEMPORAL_EVENTS_CSV = FEATURES_DIR / "temporal_events.csv"
TEMPORAL_PAUSES_CSV = FEATURES_DIR / "temporal_within_turn_pauses.csv"
TEMPORAL_REPORT_CSV = REPORTS_DIR / "temporal_processing_report.csv"
TEMPORAL_REPORT_MD = REPORTS_DIR / "temporal_analysis_report.md"


def load_manifest() -> dict:
    with open(MANIFEST_CSV, newline="") as f:
        return {row["anon_id"]: row for row in csv.DictReader(f)}


def load_acoustic_recording_level() -> pd.DataFrame:
    """anon_id -> duration_s (one row per anon_id; acoustic table has an
    'original'/'denoised' pair per recording, we only need duration once)."""
    df = pd.read_csv(RECORDING_LEVEL_ACOUSTIC_CSV)
    df = df[df["version"] == "original"][["anon_id", "duration_s"]]
    return df.set_index("anon_id")["duration_s"].to_dict()


def load_original_timeseries() -> dict:
    """
    Loads outputs/features/timeseries.csv (original version only, since
    within-turn-pause detection works on the untouched signal) and
    groups it by anon_id -> (times ndarray, rms ndarray), sorted by time.
    """
    df = pd.read_csv(TIMESERIES_CSV, usecols=["anon_id", "version", "time", "rms"])
    df = df[df["version"] == "original"]
    grouped = {}
    for anon_id, g in df.groupby("anon_id", sort=False):
        g = g.sort_values("time")
        grouped[anon_id] = (g["time"].to_numpy(), g["rms"].to_numpy())
    return grouped


def build_turn_events(anon_id: str, label: str, seq: list) -> list:
    """One row per turn (either channel) — event_type='turn'. gap_before /
    gap_after and previous_speaker/next_speaker together capture speaker
    transitions and response latency (a target turn with
    previous_speaker='other' has gap_before == its response latency), so
    a separate 'transition'/'response' event table would just duplicate
    this one under a different name."""
    events = []
    n = len(seq)
    for i, t in enumerate(seq):
        prev_t = seq[i - 1] if i > 0 else None
        next_t = seq[i + 1] if i < n - 1 else None
        events.append({
            "anon_id": anon_id,
            "label": label,
            "event_type": "turn",
            "speaker": "target" if t["channel"] == 0 else "other",
            "start": t["start"],
            "end": t["end"],
            "duration": t["end"] - t["start"],
            "previous_speaker": (None if prev_t is None else ("target" if prev_t["channel"] == 0 else "other")),
            "next_speaker": (None if next_t is None else ("target" if next_t["channel"] == 0 else "other")),
            "gap_before": (None if prev_t is None else t["start"] - prev_t["end"]),
            "gap_after": (None if next_t is None else next_t["start"] - t["end"]),
        })
    return events


def process_one(anon_id: str, manifest_row: dict, duration_s: float, ts_lookup: dict):
    turns, val_report = load_and_validate_turns(anon_id, duration_s)
    if val_report["status"] != "valid" or not turns:
        return None, val_report

    target_turns = [t for t in turns if t["channel"] == 0]
    other_turns = [t for t in turns if t["channel"] == 1]
    seq = tf.build_merged_sequence(turns)

    row = {
        "anon_id": anon_id,
        "label": manifest_row["label"],
        "split": manifest_row.get("split", ""),
        "duration_s": duration_s,
        "has_turn_data": True,
    }
    row.update(tf.target_turn_stats(target_turns))
    row.update(tf.other_turn_stats(other_turns))
    row.update(tf.speaking_ratio_stats(target_turns, turns, duration_s))

    trans_stats, gaps = tf.transition_stats(seq)
    row.update(trans_stats)

    resp_stats, latencies = tf.response_latency_stats(seq)
    row.update(resp_stats)

    times, rms = ts_lookup.get(anon_id, (np.array([]), np.array([])))
    pauses = tf.detect_within_turn_pauses(target_turns, times, rms)
    row.update(tf.within_turn_pause_summary(pauses, row["target_total_speaking_time"]))

    row.update(tf.variability_stats(target_turns, latencies, gaps))

    events = build_turn_events(anon_id, manifest_row["label"], seq)
    pause_rows = [
        {"anon_id": anon_id, "label": manifest_row["label"], **p}
        for p in pauses
    ]

    return {"row": row, "events": events, "pauses": pause_rows}, val_report


def run():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    manifest = load_manifest()
    duration_by_id = load_acoustic_recording_level()

    print("Validating all JSON turn files...")
    val_counts, val_per_file = validate_all(duration_by_id)
    print(val_counts)

    print("Loading original Channel-0 time-series for pause detection...")
    ts_lookup = load_original_timeseries()

    recording_rows = []
    all_events = []
    all_pauses = []
    failures = []
    processed = 0
    skipped_empty = 0

    audio_ids = sorted(duration_by_id.keys())
    for anon_id in audio_ids:
        manifest_row = manifest.get(anon_id)
        if manifest_row is None:
            failures.append({"anon_id": anon_id, "reason": "not in manifest.csv"})
            continue

        report = val_per_file[anon_id]
        if report["status"] == "empty":
            skipped_empty += 1
            continue
        if report["status"] in ("missing_file", "parse_error"):
            failures.append({"anon_id": anon_id, "reason": f"turn json {report['status']}"})
            continue

        try:
            result, _ = process_one(anon_id, manifest_row, duration_by_id[anon_id], ts_lookup)
        except Exception as e:
            failures.append({"anon_id": anon_id, "reason": f"{type(e).__name__}: {e}"})
            continue

        if result is None:
            failures.append({"anon_id": anon_id, "reason": "no valid turns after validation"})
            continue

        recording_rows.append(result["row"])
        all_events.extend(result["events"])
        all_pauses.extend(result["pauses"])
        processed += 1
        print(f"[{processed}] ok {anon_id}")

    rec_df = pd.DataFrame(recording_rows)
    rec_df.to_csv(TEMPORAL_RECORDING_LEVEL_CSV, index=False)

    events_df = pd.DataFrame(all_events)
    events_df.to_csv(TEMPORAL_EVENTS_CSV, index=False)

    pauses_df = pd.DataFrame(all_pauses)
    pauses_df.to_csv(TEMPORAL_PAUSES_CSV, index=False)

    with open(TEMPORAL_REPORT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["anon_id", "status", "reason"])
        for row in recording_rows:
            w.writerow([row["anon_id"], "ok", ""])
        for entry in failures:
            w.writerow([entry["anon_id"], "failed", entry["reason"]])

    elapsed = time.time() - t0
    report_md = f"""# Temporal / conversational analysis — processing report

- Expected recordings (total in files/audio): {len(audio_ids)}
- Recordings with valid turn metadata processed: {processed}
- Recordings skipped (empty JSON, no fabricated turns): {skipped_empty}
- Failures: {len(failures)}
- Wall time: {elapsed:.1f}s

## JSON validation (temporal.turns_io.validate_all)

- Total JSON files checked: {val_counts['total_json_files']}
- Valid turn files: {val_counts['valid_turn_files']}
- Empty JSON files: {val_counts['empty_json_files']}
- Invalid (missing/unparseable) JSON files: {val_counts['invalid_json_files']}
- Files with invalid timestamps (dropped individual turns): {val_counts['files_with_invalid_timestamps']}
- Files with same-channel overlapping turns: {val_counts['files_with_overlapping_turns']}
- Files with unexpected channel values: {val_counts['files_with_unexpected_channel']}
- Total individual turns dropped for validation problems: {val_counts['total_dropped_turns']}

## Settings

- target_speaking_ratio denominator: conversation span (first turn start -> last turn end, both channels), NOT wav duration_s
- near-zero gap/response-latency tolerance: {tf.NEAR_ZERO_GAP_TOLERANCE_S}s
- short/long target turn thresholds: <{tf.SHORT_TURN_THRESHOLD_S}s / >{tf.LONG_TURN_THRESHOLD_S}s
- within-turn pause threshold: RMS below the {tf.PAUSE_RMS_PERCENTILE}th percentile of THIS recording's own in-target-turn RMS values, sustained >= {tf.MIN_PAUSE_DURATION_S}s
- within-turn pauses computed on the ORIGINAL (not denoised) Channel-0 signal

## Failures

{chr(10).join(f"- {e['anon_id']}: {e['reason']}" for e in failures) if failures else "(none)"}
"""
    with open(TEMPORAL_REPORT_MD, "w") as f:
        f.write(report_md)

    print(report_md)
    return {
        "processed": processed,
        "skipped_empty": skipped_empty,
        "failed": len(failures),
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    run()
