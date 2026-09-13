"""
Transcription pipeline: Channel 0 and Channel 1, separately, for every
recording. Local/offline only (faster-whisper, see nlp/asr.py) — no
audio leaves this machine.

For the 258 recordings with valid turn metadata (reusing
temporal.turns_io — not reimplemented): transcribe each turn
separately, preserving turn boundaries and chronological order.
For the 95 recordings with empty JSON: transcribe each channel as one
whole-recording segment; NOT split into turns.csv rows and NOT used
for turn-aligned analysis, per rules.md / the task instructions.

Speaker typing (never "context" for Channel 1):
  channel 0 -> speaker_type = manifest label ("human"/"synthetic"), label_source = "manifest.csv"
  channel 1 -> speaker_type = "confirmed_ai_agent",                 label_source = "dataset_specification"

Writes outputs/transcripts/recordings.csv (one row per anon_id x channel)
and outputs/transcripts/turns.csv (one row per turn, valid-turn
recordings only). Never touches files/ or any existing acoustic/temporal
output.

Run with: python -m nlp.transcribe_pipeline
"""
import csv
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.audio_io import MANIFEST_CSV, load_stereo
from preprocessing.pipeline import FEATURES_DIR as ACOUSTIC_FEATURES_DIR
from temporal.turns_io import load_and_validate_turns
from nlp.asr import get_model, resample_to_16k, transcribe_array, TARGET_SR

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS_DIR = REPO_ROOT / "outputs" / "transcripts"
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"
RECORDINGS_CSV = TRANSCRIPTS_DIR / "recordings.csv"
TURNS_CSV = TRANSCRIPTS_DIR / "turns.csv"
TRANSCRIBE_REPORT_MD = REPORTS_DIR / "transcription_report.md"
TRANSCRIBE_REPORT_CSV = REPORTS_DIR / "transcription_report.csv"


def load_manifest() -> dict:
    with open(MANIFEST_CSV, newline="") as f:
        return {row["anon_id"]: row for row in csv.DictReader(f)}


def load_acoustic_duration_by_id() -> dict:
    df = pd.read_csv(ACOUSTIC_FEATURES_DIR / "recording_level.csv")
    df = df[df["version"] == "original"][["anon_id", "duration_s"]]
    return df.set_index("anon_id")["duration_s"].to_dict()


def _speaker_type_and_source(channel: int, manifest_row: dict):
    if channel == 0:
        return manifest_row["label"], "manifest.csv"
    return "confirmed_ai_agent", "dataset_specification"


def _init_worker():
    get_model()  # load once per worker process, reused for every task


def process_one(args):
    anon_id, manifest_row, duration_s = args
    try:
        ch0, ch1, sr = load_stereo(anon_id)
    except Exception as e:
        return {"anon_id": anon_id, "status": "failed", "reason": f"{type(e).__name__}: {e}"}

    ch0_16k = resample_to_16k(ch0, sr)
    ch1_16k = resample_to_16k(ch1, sr)
    channel_audio = {0: ch0_16k, 1: ch1_16k}

    turns, val_report = load_and_validate_turns(anon_id, duration_s)
    has_turn_data = val_report["status"] == "valid" and len(turns) > 0

    recording_rows = []
    turn_rows = []

    if has_turn_data:
        turns_sorted = sorted(turns, key=lambda t: t["start"])
        per_channel_turns = {0: [], 1: []}
        for t in turns_sorted:
            per_channel_turns[t["channel"]].append(t)

        for channel in (0, 1):
            speaker_type, label_source = _speaker_type_and_source(channel, manifest_row)
            texts = []
            logprobs = []
            no_speech_probs = []
            total_dur = 0.0
            for idx, t in enumerate(per_channel_turns[channel]):
                start_sample = int(t["start"] * TARGET_SR)
                end_sample = int(t["end"] * TARGET_SR)
                seg = channel_audio[channel][start_sample:end_sample]
                result = transcribe_array(seg)
                dur = t["end"] - t["start"]
                total_dur += dur
                texts.append(result["text"])
                if not np.isnan(result["avg_logprob"]):
                    logprobs.append(result["avg_logprob"])
                    no_speech_probs.append(result["no_speech_prob"])
                turn_rows.append({
                    "anon_id": anon_id,
                    "channel": channel,
                    "speaker_type": speaker_type,
                    "label": manifest_row["label"],
                    "label_source": label_source,
                    "turn_index": idx,
                    "start": t["start"],
                    "end": t["end"],
                    "duration": dur,
                    "text": result["text"],
                    "asr_model": f"faster-whisper-small-int8",
                    "avg_logprob": result["avg_logprob"],
                    "no_speech_prob": result["no_speech_prob"],
                    "language": result["language"],
                    "language_probability": result["language_probability"],
                })

            recording_rows.append({
                "anon_id": anon_id,
                "channel": channel,
                "speaker_type": speaker_type,
                "label": manifest_row["label"],
                "split": manifest_row.get("split", ""),
                "label_source": label_source,
                "duration_s": duration_s,
                "has_turn_data": True,
                "transcription_status": "turn_level",
                "n_turns": len(per_channel_turns[channel]),
                "transcribed_speaking_time_s": total_dur,
                "full_text": " ".join(t for t in texts if t),
                "asr_model": "faster-whisper-small-int8",
                "mean_avg_logprob": float(np.mean(logprobs)) if logprobs else np.nan,
                "mean_no_speech_prob": float(np.mean(no_speech_probs)) if no_speech_probs else np.nan,
            })
    else:
        for channel in (0, 1):
            speaker_type, label_source = _speaker_type_and_source(channel, manifest_row)
            result = transcribe_array(channel_audio[channel])
            recording_rows.append({
                "anon_id": anon_id,
                "channel": channel,
                "speaker_type": speaker_type,
                "label": manifest_row["label"],
                "split": manifest_row.get("split", ""),
                "label_source": label_source,
                "duration_s": duration_s,
                "has_turn_data": False,
                "transcription_status": "whole_recording",
                "n_turns": 0,
                "transcribed_speaking_time_s": duration_s,
                "full_text": result["text"],
                "asr_model": "faster-whisper-small-int8",
                "mean_avg_logprob": result["avg_logprob"],
                "mean_no_speech_prob": result["no_speech_prob"],
            })

    return {"anon_id": anon_id, "status": "ok", "recording_rows": recording_rows, "turn_rows": turn_rows}


def run(n_workers: int = 6, limit: int = None):
    """
    Writes recordings.csv / turns.csv INCREMENTALLY (one recording's rows
    flushed to disk as soon as it's done), not accumulated in memory and
    written once at the end — this job can run for hours, and we don't
    want an interruption to lose everything transcribed so far.
    """
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    duration_by_id = load_acoustic_duration_by_id()
    audio_ids = sorted(duration_by_id.keys())
    if limit:
        audio_ids = audio_ids[:limit]

    tasks = [(aid, manifest[aid], duration_by_id[aid]) for aid in audio_ids if aid in manifest]
    missing_manifest = [aid for aid in audio_ids if aid not in manifest]

    t0 = time.time()
    failures = [{"anon_id": a, "reason": "not in manifest.csv"} for a in missing_manifest]
    n_ok = 0
    n_turn_level = 0
    n_whole = 0
    n_recording_rows = 0
    n_turn_rows = 0

    rec_writer = None
    turn_writer = None

    with open(RECORDINGS_CSV, "w", newline="") as f_rec, \
         open(TURNS_CSV, "w", newline="") as f_turn, \
         mp.Pool(processes=n_workers, initializer=_init_worker) as pool:

        for i, result in enumerate(pool.imap_unordered(process_one, tasks), start=1):
            if result["status"] != "ok":
                failures.append({"anon_id": result["anon_id"], "reason": result["reason"]})
                print(f"[{i}/{len(tasks)}] FAILED {result['anon_id']}: {result['reason']}", flush=True)
                continue

            for row in result["recording_rows"]:
                if rec_writer is None:
                    rec_writer = csv.DictWriter(f_rec, fieldnames=list(row.keys()))
                    rec_writer.writeheader()
                rec_writer.writerow(row)
                n_recording_rows += 1
            for row in result["turn_rows"]:
                if turn_writer is None:
                    turn_writer = csv.DictWriter(f_turn, fieldnames=list(row.keys()))
                    turn_writer.writeheader()
                turn_writer.writerow(row)
                n_turn_rows += 1
            f_rec.flush()
            f_turn.flush()

            n_ok += 1
            status = result["recording_rows"][0]["transcription_status"] if result["recording_rows"] else "?"
            if status == "turn_level":
                n_turn_level += 1
            else:
                n_whole += 1
            print(f"[{i}/{len(tasks)}] ok {result['anon_id']} ({status})", flush=True)

    with open(TRANSCRIBE_REPORT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["anon_id", "status", "reason"])
        for aid in [t[0] for t in tasks if t[0] not in {e["anon_id"] for e in failures}]:
            w.writerow([aid, "ok", ""])
        for e in failures:
            w.writerow([e["anon_id"], "failed", e["reason"]])

    elapsed = time.time() - t0
    report = f"""# Transcription report

- Recordings expected: {len(audio_ids)}
- Recordings transcribed successfully: {n_ok}
  - turn-level (valid JSON turn metadata): {n_turn_level}
  - whole-recording (empty JSON): {n_whole}
- Failures: {len(failures)}
- recordings.csv rows: {n_recording_rows} (2 per successful recording: channel 0 + channel 1)
- turns.csv rows: {n_turn_rows}
- Wall time: {elapsed:.1f}s ({n_workers} worker processes)

## ASR settings

- Model: faster-whisper "small", compute_type=int8, CPU, local/offline (no cloud API)
- Language forced to Spanish ("es")
- Audio resampled from 8000 Hz to 16000 Hz (faster-whisper's expected rate) before transcription
- Turn-level recordings: each turn transcribed as its own segment, in chronological order
- Whole-recording (empty-JSON) recordings: entire channel transcribed as one segment

## Failures

{chr(10).join(f"- {e['anon_id']}: {e['reason']}" for e in failures) if failures else "(none)"}
"""
    with open(TRANSCRIBE_REPORT_MD, "w") as f:
        f.write(report)
    print(report)
    return {"ok": n_ok, "failed": len(failures), "elapsed_s": elapsed}


if __name__ == "__main__":
    run()
