"""DIAGNOSTIC ONLY -- finds the minimum raw-sample accumulation window
that removes the chunked-resampling artifact, before touching production
code. Not imported by anything; safe to delete after use."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.audio_utils import TARGET_SR, HOP_SECONDS, log_mel_spectrogram, to_mono_resampled
from app.live_infer import RollingDetector, default_model_path
from app.model import load_model
from preprocessing.normalize import normalize_rms

AUDIO_DIR = Path(__file__).resolve().parents[1] / "files" / "audio"
MIC_SR = 48000
MIC_CHUNK = 4096

TEST_FILES = [
    ("call_0294f969f98b", "synthetic"),
    ("call_0847d7417bb1", "synthetic"),
]


def load_native(anon_id):
    y, sr = sf.read(AUDIO_DIR / f"{anon_id}.wav", always_2d=True)
    y = y[:, 0].astype(np.float64)
    assert sr == TARGET_SR
    return y


def run_file_mode(model, y, sr):
    detector = RollingDetector(model)
    chunk_len = int(HOP_SECONDS * sr)
    windows, consumed = [], 0
    for start in range(0, len(y), chunk_len):
        chunk = y[start:start + chunk_len]
        if len(chunk) == 0:
            break
        detector.push(chunk, sr)
        consumed += len(chunk)
        if detector.ready():
            windows.append({"t": consumed / sr, "buffer": detector.buffer.copy(),
                             **detector.step()})
    return windows


def run_mic_mode_buffered(model, y_mic, mic_sr, ws_chunk, accumulate_seconds):
    """Simulates the PROPOSED fix: raw samples arrive in small ws_chunk
    pieces (like the real browser), but push() only calls resample_poly
    once `accumulate_seconds` worth of raw audio has piled up."""
    detector = RollingDetector(model)
    raw_pending = np.zeros(0, dtype=np.float64)
    min_needed = int(accumulate_seconds * mic_sr)
    windows, consumed = [], 0
    for start in range(0, len(y_mic), ws_chunk):
        chunk = y_mic[start:start + ws_chunk].astype(np.float32)
        if len(chunk) == 0:
            break
        raw_pending = np.concatenate([raw_pending, chunk.astype(np.float64)])
        consumed += len(chunk)
        if len(raw_pending) >= min_needed:
            mono = to_mono_resampled(raw_pending, mic_sr)
            raw_pending = np.zeros(0, dtype=np.float64)
            n = len(mono)
            if n >= 24000:
                detector.buffer = mono[-24000:]
            else:
                detector.buffer = np.concatenate([detector.buffer[n:], mono])
            detector.filled = min(detector.filled + n, 24000)
            detector.since_last_score += n
            if detector.ready():
                windows.append({"t": consumed / mic_sr, "buffer": detector.buffer.copy(),
                                 **detector.step()})
    return windows


def compare(ref_windows, other_windows):
    score_diffs, silence_mismatches, mel_diffs = [], 0, []
    for ref in ref_windows:
        best = min(other_windows, key=lambda w: abs(w["t"] - ref["t"]))
        if abs(best["t"] - ref["t"]) > HOP_SECONDS * 0.6:
            continue
        if ref["silence"] != best["silence"]:
            silence_mismatches += 1
        if ref["raw"] is not None and best["raw"] is not None:
            score_diffs.append(abs(ref["raw"] - best["raw"]))
        ref_norm, _ = normalize_rms(ref["buffer"])
        oth_norm, _ = normalize_rms(best["buffer"])
        mel_diffs.append(float(np.max(np.abs(log_mel_spectrogram(ref_norm) - log_mel_spectrogram(oth_norm)))))
    return score_diffs, silence_mismatches, mel_diffs


def main():
    model = load_model(default_model_path())
    for accumulate_seconds in [0.085, 0.25, 0.5, 1.0, 2.0]:
        print(f"\n--- accumulate_seconds={accumulate_seconds} ---")
        for anon_id, label in TEST_FILES:
            y_native = load_native(anon_id)
            y_mic = librosa.resample(y_native, orig_sr=TARGET_SR, target_sr=MIC_SR, res_type="soxr_hq")
            file_windows = run_file_mode(model, y_native, TARGET_SR)
            mic_windows = run_mic_mode_buffered(model, y_mic, MIC_SR, MIC_CHUNK, accumulate_seconds)
            score_diffs, silence_mismatches, mel_diffs = compare(file_windows, mic_windows)
            print(f"  {anon_id} ({label}): n_file={len(file_windows)} n_mic={len(mic_windows)} "
                  f"silence_mismatches={silence_mismatches} "
                  f"score_diff median={np.median(score_diffs):.4f} mean={np.mean(score_diffs):.4f} max={np.max(score_diffs):.4f}")


if __name__ == "__main__":
    main()
