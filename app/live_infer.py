"""
Real-time AI-voice-likelihood detector. Uses the EXACT SAME preprocessing
(app/audio_utils.py) and model (app/model.py) as training
(modeling/train_cnn.py) -- no separate/re-derived logic.

Two ways to run:
  1. Live microphone (needs a working input device + `sounddevice`):
       python -m app.live_infer
  2. Simulated live input from an existing wav file (useful for testing
     without a microphone, or for a reproducible demo):
       python -m app.live_infer --file path/to/some.wav

Both paths feed audio into the same RollingDetector, which emits one
updated AI-likelihood score every HOP_SECONDS (50%-overlapping 3s
windows), exactly matching the training window/hop configuration.
"""
import argparse
import queue
import sys
import time

import numpy as np
import soundfile as sf
import torch

from app.audio_utils import (
    TARGET_SR, WINDOW_SAMPLES, HOP_SAMPLES, HOP_SECONDS,
    to_mono_resampled, log_mel_spectrogram,
)
from preprocessing.normalize import normalize_rms
from app.model import load_model
from app.decision import DecisionEngine, SILENCE_RMS_THRESHOLD
from app.evidence import EvidenceSession, compute_window_features, compute_window_f0
from app.speaker_guard import SpeakerGuard

MODEL_PATH = None  # set by caller / resolved lazily below


def default_model_path():
    from pathlib import Path
    import os
    override = os.environ.get("CNN_MODEL_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "outputs" / "models" / "cnn_live_detector.pt"


class RollingDetector:
    """
    Maintains a rolling ring buffer of the last WINDOW_SAMPLES of audio
    (already resampled to TARGET_SR). Call `push(samples, sr)` with new
    audio as it arrives; call `ready()`/`step()` once enough audio has
    accumulated since the last scored window (HOP_SAMPLES worth).

    `step()` never raises: a single malformed/invalid window (NaN, a
    transient decode glitch, etc.) is caught and reported via the
    result's `error` field instead of crashing a live session.
    """

    def __init__(self, model, warm_up: bool = True):
        self.model = model
        self.buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float64)
        self.filled = 0          # how many real (non-zero-padded) samples we've ever seen
        self.since_last_score = 0
        self.decision = DecisionEngine()
        self.evidence = EvidenceSession()
        self.guard = SpeakerGuard()
        if warm_up:
            # librosa's first mel/FFT call and torch's first forward pass
            # both pay a one-time ~500-600ms cold-start cost (thread pool
            # / cache warm-up). Pay it once here, at construction, rather
            # than on the first real speech window during a live demo.
            dummy = np.random.RandomState(0).randn(WINDOW_SAMPLES) * 0.05
            mel = log_mel_spectrogram(dummy)
            with torch.no_grad():
                self.model(torch.from_numpy(mel).unsqueeze(0).unsqueeze(0))

    def push(self, samples: np.ndarray, sr: int):
        mono = to_mono_resampled(samples, sr)
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(mono)
        if n >= WINDOW_SAMPLES:
            self.buffer = mono[-WINDOW_SAMPLES:]
        else:
            self.buffer = np.concatenate([self.buffer[n:], mono])
        self.filled = min(self.filled + n, WINDOW_SAMPLES)
        self.since_last_score += n

    def ready(self) -> bool:
        return self.filled >= WINDOW_SAMPLES and self.since_last_score >= HOP_SAMPLES

    def step(self) -> dict:
        """
        Runs one full scoring cycle on the CURRENT buffer:
          1. silence check (unchanged)
          2. acoustic features + F0 (always computed for non-silent
             windows -- needed by the speaker-change guard regardless of
             whether the CNN ends up running this window)
          3. speaker-change guard (app/speaker_guard.py): "normal" ->
             proceed as before; "speaker_change"/"stabilizing" -> the
             CNN is NOT run and evidence is NOT updated this window,
             per the task requirement that a new/unstable speaker must
             not contaminate the running evidence.
          4. (normal phase only) log-Mel -> CNN -> sigmoid -> smoothing/
             hysteresis decision layer (app/decision.py)

        Returns a dict with raw/smoothed score, displayed label, and
        `silence`/`speaker_change`/`stabilizing` flags (all False except
        the one that applies), a latency breakdown (ms), and `error`
        (None on success). Never raises -- a single malformed/invalid
        window is caught and reported via `error` instead.
        """
        t0 = time.perf_counter()
        self.since_last_score = 0
        t_start = self.evidence.next_t_start()  # deterministic: window_index * HOP_SECONDS
        try:
            raw_rms = float(np.sqrt(np.mean(np.square(self.buffer))))
            if raw_rms < SILENCE_RMS_THRESHOLD:
                result = self.decision.update(None, is_silence=True)
                self.evidence.add_silence(t_start)
                return {**result, "speaker_change": False, "stabilizing": False,
                        "feature_ms": None, "preprocess_ms": None, "inference_ms": None,
                        "total_ms": (time.perf_counter() - t0) * 1000, "error": None}

            y_norm, _ = normalize_rms(self.buffer)
            tf0 = time.perf_counter()
            features = compute_window_features(self.buffer, y_norm, TARGET_SR)
            f0_median = compute_window_f0(self.buffer, TARGET_SR)
            feature_ms = (time.perf_counter() - tf0) * 1000

            guard_result = self.guard.process(f0_median, features)
            phase = guard_result["phase"]

            if phase == "speaker_change":
                self.evidence.add_speaker_change(t_start)
                self.decision.reset_for_new_speaker()
                return {"raw": None, "smoothed": None, "label": "Speaker change detected",
                        "silence": False, "speaker_change": True, "stabilizing": False,
                        "feature_ms": feature_ms, "preprocess_ms": None, "inference_ms": None,
                        "total_ms": (time.perf_counter() - t0) * 1000, "error": None}

            if phase == "stabilizing":
                self.evidence.add_stabilizing(t_start)
                return {"raw": None, "smoothed": None, "label": "Stabilizing",
                        "silence": False, "speaker_change": False, "stabilizing": True,
                        "feature_ms": feature_ms, "preprocess_ms": None, "inference_ms": None,
                        "total_ms": (time.perf_counter() - t0) * 1000, "error": None}

            t1 = time.perf_counter()
            mel = log_mel_spectrogram(y_norm)
            t2 = time.perf_counter()
            x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,n_frames)
            with torch.no_grad():
                prob = torch.sigmoid(self.model(x)).item()
            t3 = time.perf_counter()

            result = self.decision.update(prob, is_silence=False)
            self.evidence.add_speech(t_start, result["raw"], result["smoothed"], result["label"], features, f0_median)
            return {**result, "speaker_change": False, "stabilizing": False,
                    "feature_ms": feature_ms,
                    "preprocess_ms": (t2 - t1) * 1000,
                    "inference_ms": (t3 - t2) * 1000,
                    "total_ms": (t3 - t0) * 1000,
                    "error": None}
        except Exception as e:
            return {"raw": None, "smoothed": None, "label": "Error", "silence": False,
                     "speaker_change": False, "stabilizing": False,
                     "feature_ms": None, "preprocess_ms": None, "inference_ms": None,
                     "total_ms": (time.perf_counter() - t0) * 1000,
                     "error": f"{type(e).__name__}: {e}"}

    def get_report(self) -> dict:
        """The full AI Voice Analysis Report for everything analyzed so
        far this session (see app/evidence.py)."""
        return self.evidence.summary(self.decision.label)


def _print_step(prefix: str, result: dict):
    if result["error"]:
        print(f"{prefix}[error] {result['error']}")
        return
    if result["silence"]:
        print(f"{prefix}(silence)")
        return
    if result["speaker_change"]:
        print(f"{prefix}*** SPEAKER CHANGE DETECTED ***")
        return
    if result["stabilizing"]:
        print(f"{prefix}(stabilizing new speaker...)")
        return
    bar = "#" * int(result["smoothed"] * 40)
    print(f"{prefix}raw={result['raw']:.3f} smoothed={result['smoothed']:.3f} "
          f"[{result['label']:<11}] |{bar:<40}| "
          f"({result['total_ms']:.0f}ms/window)")


def run_from_file(path: str, model, realtime: bool = False):
    """
    Simulates live streaming by feeding a wav file in HOP_SECONDS-sized
    chunks -- the SAME RollingDetector.step() path used by the
    microphone and the FastAPI demo, processed incrementally (never one
    whole-file prediction). Useful as a fallback demo if microphone
    permissions fail during the presentation.
    `realtime=True` paces chunks to real wall-clock time (matches how a
    live mic session would feel); default is fast playback for quicker
    iteration while testing.
    """
    y, sr = sf.read(path, always_2d=True)
    y = y[:, 0].astype(np.float64)
    chunk_len = int(HOP_SECONDS * sr)
    detector = RollingDetector(model)
    print(f"Simulating live input from {path} (sr={sr}Hz, {len(y)/sr:.1f}s total)")
    print(f"Emitting a score every {HOP_SECONDS:.1f}s (3s window, 50% overlap)\n")
    for start in range(0, len(y), chunk_len):
        chunk = y[start:start + chunk_len]
        if len(chunk) == 0:
            break
        detector.push(chunk, sr)
        if detector.ready():
            result = detector.step()
            _print_step(f"[{start/sr:6.1f}s] ", result)
        time.sleep(HOP_SECONDS if realtime else HOP_SECONDS * 0.05)


def run_from_microphone(model, device_sr: int = 16000):
    try:
        import sounddevice as sd
    except Exception as e:
        print(f"Microphone unavailable: could not import sounddevice ({e}). "
              f"Use --file <path.wav> instead.")
        sys.exit(1)

    detector = RollingDetector(model)
    audio_q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    print(f"Listening on the default microphone at {device_sr}Hz...")
    print(f"Emitting a score every {HOP_SECONDS:.1f}s (3s window, 50% overlap)")
    print("Press Ctrl+C to stop.\n")

    try:
        stream = sd.InputStream(samplerate=device_sr, channels=1, callback=callback)
    except Exception as e:
        print(f"Microphone unavailable ({type(e).__name__}: {e}). "
              f"Use --file <path.wav> instead to demo without a mic.")
        sys.exit(1)

    with stream:
        try:
            while True:
                chunk = audio_q.get()
                detector.push(chunk, device_sr)
                if detector.ready():
                    result = detector.step()
                    _print_step("", result)
        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="Live AI-voice-likelihood detector")
    parser.add_argument("--file", type=str, default=None,
                         help="Simulate live input from this wav file instead of the microphone")
    parser.add_argument("--model", type=str, default=None,
                         help="Path to trained CNN weights (default: outputs/models/cnn_live_detector.pt)")
    parser.add_argument("--realtime", action="store_true",
                         help="Pace --file playback to real wall-clock time (default: fast)")
    args = parser.parse_args()

    model_path = args.model or default_model_path()
    if not str(model_path) or not __import__("pathlib").Path(model_path).exists():
        print(f"Model weights not found at {model_path}. Run `python -m modeling.train_cnn` first.")
        sys.exit(1)
    model = load_model(model_path)

    if args.file:
        run_from_file(args.file, model, realtime=args.realtime)
    else:
        run_from_microphone(model)


if __name__ == "__main__":
    main()
