"""
DIAGNOSTIC ONLY -- not production code, not imported by anything.

Question: production model works on file-mode inference but under-detects
AI voices through the live microphone. Model/weights/thresholds are
EXPLICITLY OUT OF SCOPE for this script -- it only inspects and compares
the audio preprocessing that feeds the (identical, untouched) CNN.

Both file mode (app/main.py:_file_loop, app/live_infer.py:run_from_file)
and the browser mic mode (app/main.py:ws_mic) push audio into the SAME
RollingDetector.push()/step(). The only structural difference between
them is *how* push() is called:

  FILE MODE:  chunk = HOP_SECONDS * file_sr samples, file_sr == TARGET_SR
              (8000Hz -- the project's recordings are already 8kHz
              telephone-band audio) => to_mono_resampled() takes the
              `sr != TARGET_SR` branch's `False` path and NEVER calls
              resample_poly at all.

  MIC MODE:   browser ScriptProcessorNode(4096, 1, 1) delivers 4096-sample
              chunks at audioCtx.sampleRate (typically 44100 or 48000Hz,
              NOT 8000) over the WebSocket. Every single 4096-sample chunk
              is independently passed to push() -> to_mono_resampled() ->
              scipy.signal.resample_poly(chunk, 8000, sr). resample_poly
              is STATELESS: it has no memory of the previous call, so its
              anti-aliasing FIR filter implicitly zero-pads at the start
              and end of *each* short chunk. Doing this once per ~85ms
              chunk, continuously, for an entire live session is a classic
              "chunked streaming resample without filter continuity" bug
              pattern -- worth measuring directly rather than assuming.

This script does NOT have access to a physical microphone (sandboxed
environment). It cannot record real room/ADC audio. What it CAN do,
using only the real, unmodified production code
(app.audio_utils.to_mono_resampled, app.live_infer.RollingDetector), is
isolate the SOFTWARE preprocessing question precisely:

  Take a real recording already at the model's native 8kHz. Upsample it
  ONCE (clean, high-quality, single-shot) to a typical browser
  AudioContext rate (48000Hz) -- this recovers "what a mic capturing this
  exact same speech at 48kHz would hand the browser" (telephone-band
  8kHz-limited content loses nothing being represented at a higher rate).
  Then feed that 48kHz signal into the REAL RollingDetector two ways:

    A. FILE-MODE emulation: large chunks, sr already == TARGET_SR (the
       original 8kHz file, unchanged) -- this is exactly run_from_file's
       real code path.
    B. MIC-MODE emulation: the SAME audio content, but as the 48kHz
       upsampled version, fed in 4096-sample chunks at sr=48000 -- this
       is exactly ws_mic's real code path (same push() call, same sr
       plumbing, same chunk size).

  Compare the resulting ring-buffer contents, log-Mel tensors, and raw
  CNN scores at matching time windows. If B diverges from A while both
  are fed the *same underlying speech*, the divergence is attributable to
  the mic path's chunked resampling, not to the model or the content.

  A third arm (C) resamples the SAME 48kHz signal to 8kHz in one single
  whole-signal call (no chunking) -- this isolates whether resampling
  ITSELF causes drift, vs. specifically the chunk-by-chunk repetition of
  it.

No production file is imported for its side effects beyond normal
inference calls. Nothing here writes to outputs/models/ or touches
app/decision.py's constants.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio_utils import (
    TARGET_SR, WINDOW_SAMPLES, HOP_SAMPLES, HOP_SECONDS, log_mel_spectrogram,
)
from app.live_infer import RollingDetector, default_model_path
from app.model import load_model

AUDIO_DIR = Path(__file__).resolve().parents[1] / "files" / "audio"
REPORT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MIC_SR = 48000          # typical browser AudioContext.sampleRate
MIC_CHUNK = 4096         # app/main.py: audioCtx.createScriptProcessor(4096, 1, 1)

TEST_FILES = [
    ("call_0294f969f98b", "synthetic"),
    ("call_0847d7417bb1", "synthetic"),
    ("call_0e1e2f29bfdc", "human"),
    ("call_ca184bb77a63", "human"),
]


def load_native(anon_id):
    path = AUDIO_DIR / f"{anon_id}.wav"
    y, sr = sf.read(path, always_2d=True)
    y = y[:, 0].astype(np.float64)
    assert sr == TARGET_SR, f"{anon_id}: expected native {TARGET_SR}Hz, got {sr}Hz"
    return y, sr


def make_mic_emulated(y_native_8k):
    """One clean upsample 8kHz -> 48kHz -- simulates 'what a mic capturing
    this same speech at 48kHz would have produced' for a signal that is
    telephone-band (<=4kHz content) to begin with."""
    y_48k = librosa.resample(y_native_8k, orig_sr=TARGET_SR, target_sr=MIC_SR, res_type="soxr_hq")
    return y_48k


def run_file_mode(model, y, sr):
    """Real app/live_infer.py:run_from_file code path, verbatim logic.
    `t_elapsed` is the TRUE cumulative elapsed time of original-signal
    content consumed so far (end of the just-pushed chunk) -- NOT the
    loop variable's chunk-start index, which under-counts by a full
    chunk when chunks are large (see module docstring correction)."""
    detector = RollingDetector(model)
    chunk_len = int(HOP_SECONDS * sr)
    windows = []
    consumed = 0
    for start in range(0, len(y), chunk_len):
        chunk = y[start:start + chunk_len]
        if len(chunk) == 0:
            break
        detector.push(chunk, sr)
        consumed += len(chunk)
        if detector.ready():
            buf_snapshot = detector.buffer.copy()
            result = detector.step()
            windows.append({
                "t_start": consumed / sr,
                "buffer": buf_snapshot,
                "raw": result["raw"],
                "silence": result["silence"],
                "error": result["error"],
            })
    return windows


def run_mic_mode(model, y_mic, mic_sr, chunk_size):
    """Real app/main.py:ws_mic code path, verbatim logic: independent
    small chunks at the mic's native (non-8kHz) sample rate."""
    detector = RollingDetector(model)
    windows = []
    consumed = 0
    for start in range(0, len(y_mic), chunk_size):
        chunk = y_mic[start:start + chunk_size].astype(np.float32)
        if len(chunk) == 0:
            break
        detector.push(chunk, mic_sr)
        consumed += len(chunk)
        if detector.ready():
            buf_snapshot = detector.buffer.copy()
            result = detector.step()
            windows.append({
                "t_start": consumed / mic_sr,
                "buffer": buf_snapshot,
                "raw": result["raw"],
                "silence": result["silence"],
                "error": result["error"],
            })
    return windows


def run_oneshot_resample_mode(model, y_mic, mic_sr):
    """Control arm C: resample the WHOLE mic-rate signal to TARGET_SR in
    a single call (no chunking), then feed it exactly like file mode.
    Isolates 'resampling itself' from 'chunked resampling'."""
    from app.audio_utils import to_mono_resampled
    y_resampled_once = to_mono_resampled(y_mic, mic_sr)  # single call, whole signal
    return run_file_mode(model, y_resampled_once, TARGET_SR)


def compare_window_lists(ref_windows, other_windows, model, label):
    """Align by nearest t_start and compare buffer/tensor/score."""
    rows = []
    for ref in ref_windows:
        best = min(other_windows, key=lambda w: abs(w["t_start"] - ref["t_start"])) if other_windows else None
        if best is None or abs(best["t_start"] - ref["t_start"]) > HOP_SECONDS * 0.6:
            continue
        ref_buf, other_buf = ref["buffer"], best["buffer"]
        buf_diff = float(np.max(np.abs(ref_buf - other_buf))) if len(ref_buf) == len(other_buf) else None

        # log-mel tensors from each buffer, via the real (unmodified) function
        from preprocessing.normalize import normalize_rms
        ref_norm, _ = normalize_rms(ref_buf)
        other_norm, _ = normalize_rms(other_buf)
        ref_mel = log_mel_spectrogram(ref_norm)
        other_mel = log_mel_spectrogram(other_norm)
        mel_diff = float(np.max(np.abs(ref_mel - other_mel)))
        mel_mean_diff = float(np.mean(np.abs(ref_mel - other_mel)))

        rows.append({
            "t_start": ref["t_start"],
            "ref_silence": ref["silence"], "other_silence": best["silence"],
            "ref_raw": ref["raw"], "other_raw": best["raw"],
            "score_diff": (None if (ref["raw"] is None or best["raw"] is None)
                           else abs(ref["raw"] - best["raw"])),
            "buffer_max_abs_diff": buf_diff,
            "buffer_rms_ref": float(np.sqrt(np.mean(ref_buf**2))),
            "buffer_rms_other": float(np.sqrt(np.mean(other_buf**2))),
            "mel_max_abs_diff": mel_diff,
            "mel_mean_abs_diff": mel_mean_diff,
        })
    return rows


def main():
    model_path = default_model_path()
    print(f"Using production model: {model_path}")
    model = load_model(model_path)

    all_results = {}
    for anon_id, label in TEST_FILES:
        print(f"\n=== {anon_id} ({label}) ===")
        y_native, sr = load_native(anon_id)
        y_mic = make_mic_emulated(y_native)
        print(f"  native: {len(y_native)} samples @ {sr}Hz ({len(y_native)/sr:.1f}s)")
        print(f"  mic-emulated: {len(y_mic)} samples @ {MIC_SR}Hz ({len(y_mic)/MIC_SR:.1f}s)")

        file_windows = run_file_mode(model, y_native, sr)
        mic_windows = run_mic_mode(model, y_mic, MIC_SR, MIC_CHUNK)
        oneshot_windows = run_oneshot_resample_mode(model, y_mic, MIC_SR)

        print(f"  file-mode windows: {len(file_windows)}, "
              f"mic-mode(chunked) windows: {len(mic_windows)}, "
              f"oneshot-resample windows: {len(oneshot_windows)}")

        cmp_mic_vs_file = compare_window_lists(file_windows, mic_windows, model, "mic_vs_file")
        cmp_oneshot_vs_file = compare_window_lists(file_windows, oneshot_windows, model, "oneshot_vs_file")

        all_results[anon_id] = {
            "label": label,
            "file_scores": [w["raw"] for w in file_windows],
            "mic_chunked_scores": [w["raw"] for w in mic_windows],
            "oneshot_resample_scores": [w["raw"] for w in oneshot_windows],
            "mic_vs_file": cmp_mic_vs_file,
            "oneshot_vs_file": cmp_oneshot_vs_file,
        }

        if cmp_mic_vs_file:
            max_mel_diff = max(r["mel_max_abs_diff"] for r in cmp_mic_vs_file)
            max_score_diff = max((r["score_diff"] for r in cmp_mic_vs_file if r["score_diff"] is not None), default=None)
            print(f"  [chunked mic vs file] max mel |diff| across windows: {max_mel_diff:.3f} dB, "
                  f"max score diff: {max_score_diff}")
        if cmp_oneshot_vs_file:
            max_mel_diff2 = max(r["mel_max_abs_diff"] for r in cmp_oneshot_vs_file)
            max_score_diff2 = max((r["score_diff"] for r in cmp_oneshot_vs_file if r["score_diff"] is not None), default=None)
            print(f"  [oneshot resample vs file] max mel |diff| across windows: {max_mel_diff2:.3f} dB, "
                  f"max score diff: {max_score_diff2}")

    with open(REPORT_DIR / "_live_vs_file_pipeline_dump.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDumped full results to {REPORT_DIR / '_live_vs_file_pipeline_dump.json'}")


if __name__ == "__main__":
    main()
