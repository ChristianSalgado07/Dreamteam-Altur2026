"""
Shared preprocessing for the live CNN detector — imported by BOTH
modeling/train_cnn.py (training) and app/main.py / app/live_infer.py
(live inference), so training and inference can never drift apart
(hackathon requirement: "keep preprocessing identical between training
and inference").

Pipeline (identical for a training-set recording and a live microphone
buffer):
  1. mono float64 signal at TARGET_SR (8000 Hz -- the project's native
     rate; all existing recordings are already 8kHz telephone-band audio,
     so this is "the project's chosen model sample rate", not an
     arbitrary pick. Live mic audio is resampled down to match.)
  2. RMS-normalize (preprocessing/normalize.py:normalize_rms -- the
     SAME, already-existing, label-free, peak-safe gain used elsewhere
     in this project) so microphone loudness doesn't dominate.
  3. slice into fixed-length overlapping windows
  4. log-Mel spectrogram per window (librosa), fixed shape

No transcripts, ASR, or turn metadata anywhere in this path.
"""
from pathlib import Path

import numpy as np
import librosa
from scipy.signal import resample_poly

from preprocessing.normalize import normalize_rms

TARGET_SR = 8000          # matches the project's native recording rate
WINDOW_SECONDS = 3.0
OVERLAP = 0.5
HOP_SECONDS = WINDOW_SECONDS * (1 - OVERLAP)

WINDOW_SAMPLES = int(WINDOW_SECONDS * TARGET_SR)   # 24000
HOP_SAMPLES = int(HOP_SECONDS * TARGET_SR)          # 12000

N_MELS = 40
N_FFT = 1024      # 128ms @ 8kHz -- same frame length used throughout preprocessing/features.py
HOP_LENGTH = 256  # 32ms hop, same convention as preprocessing/features.py

# Fixed number of spectrogram frames per window, so every CNN input has
# the same shape regardless of minor rounding in librosa's framing.
N_FRAMES = 1 + (WINDOW_SAMPLES - N_FFT) // HOP_LENGTH


def to_mono_resampled(y: np.ndarray, sr: int) -> np.ndarray:
    """Mono float64 at TARGET_SR. y may already be mono (mic input) or
    stereo (loaded straight from a project wav's Channel 0 is expected
    to already be isolated by the caller -- this just handles rate)."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y[:, 0]
    if sr != TARGET_SR:
        y = resample_poly(y, TARGET_SR, sr)
    return y


def make_windows(y: np.ndarray) -> list:
    """Slice a 1D signal (already at TARGET_SR) into fixed-length,
    50%-overlapping windows. Drops a final partial window (never pads
    with fabricated silence into a real window)."""
    windows = []
    start = 0
    while start + WINDOW_SAMPLES <= len(y):
        windows.append(y[start:start + WINDOW_SAMPLES])
        start += HOP_SAMPLES
    return windows


def log_mel_spectrogram(y_window: np.ndarray) -> np.ndarray:
    """One window -> (N_MELS, N_FRAMES) log-power Mel spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=y_window, sr=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Defensive fixed-width crop/pad in case of a one-frame rounding difference
    if log_mel.shape[1] > N_FRAMES:
        log_mel = log_mel[:, :N_FRAMES]
    elif log_mel.shape[1] < N_FRAMES:
        pad = N_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="edge")
    return log_mel.astype(np.float32)


def slice_windows(y: np.ndarray, sr: int) -> list:
    """Mono/resample/normalize/window only (no log-Mel yet) -- lets a
    caller subsample cheap raw-audio windows before paying for the
    (slightly more expensive) log-Mel step. Returns raw waveform windows."""
    y = to_mono_resampled(y, sr)
    y_norm, _ = normalize_rms(y)
    return make_windows(y_norm)


def preprocess_recording(y: np.ndarray, sr: int) -> list:
    """Full pipeline for one recording/buffer -> list of (N_MELS, N_FRAMES)
    log-Mel arrays, one per window. Used identically by training (whole
    recordings) and live inference (rolling mic buffer)."""
    return [log_mel_spectrogram(w) for w in slice_windows(y, sr)]


PREPROCESSING_CONFIG = {
    "target_sr": TARGET_SR,
    "window_seconds": WINDOW_SECONDS,
    "overlap": OVERLAP,
    "hop_seconds": HOP_SECONDS,
    "n_mels": N_MELS,
    "n_fft": N_FFT,
    "hop_length": HOP_LENGTH,
    "n_frames": N_FRAMES,
    "normalization": "rms_target_0.05_peak_safety_0.99 (preprocessing/normalize.py:normalize_rms)",
}
