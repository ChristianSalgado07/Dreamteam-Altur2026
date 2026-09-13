"""
Loading and basic validation of the raw two-channel recordings.

Only reads files under files/audio — never writes back into that directory.
"""
from pathlib import Path

import numpy as np
import soundfile as sf

EXPECTED_CHANNELS = 2
EXPECTED_SAMPLE_RATE = 8000

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "files" / "audio"
JSON_DIR = REPO_ROOT / "files" / "turns" / "turns"
MANIFEST_CSV = REPO_ROOT / "files" / "csv" / "manifest.csv"


def load_channel0(anon_id: str):
    """
    Load the original stereo wav for `anon_id` and return (channel0, sample_rate).

    Raises ValueError if the file does not match the expected 2-channel,
    8000 Hz format the dataset inspection found for every recording.
    """
    path = AUDIO_DIR / f"{anon_id}.wav"
    audio, sr = sf.read(path, always_2d=True)  # shape: (n_samples, n_channels)

    if audio.shape[1] != EXPECTED_CHANNELS:
        raise ValueError(
            f"expected {EXPECTED_CHANNELS} channels, got {audio.shape[1]}"
        )
    if sr != EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"expected {EXPECTED_SAMPLE_RATE} Hz sample rate, got {sr}"
        )

    channel0 = np.asarray(audio[:, 0], dtype=np.float64)
    return channel0, sr


def load_stereo(anon_id: str):
    """
    Load both channels of the original stereo wav for `anon_id`.
    Returns (channel0, channel1, sample_rate). Same validation as
    load_channel0 — added for the transcription stage, which (unlike the
    acoustic pipeline) also needs Channel 1.
    """
    path = AUDIO_DIR / f"{anon_id}.wav"
    audio, sr = sf.read(path, always_2d=True)

    if audio.shape[1] != EXPECTED_CHANNELS:
        raise ValueError(
            f"expected {EXPECTED_CHANNELS} channels, got {audio.shape[1]}"
        )
    if sr != EXPECTED_SAMPLE_RATE:
        raise ValueError(
            f"expected {EXPECTED_SAMPLE_RATE} Hz sample rate, got {sr}"
        )

    channel0 = np.asarray(audio[:, 0], dtype=np.float64)
    channel1 = np.asarray(audio[:, 1], dtype=np.float64)
    return channel0, channel1, sr


def basic_signal_stats(y: np.ndarray, sr: int, silence_rel_threshold: float = 0.02) -> dict:
    """
    Cheap sanity-check statistics computed directly on the raw samples,
    before any transformation, to confirm the signal loaded correctly.

    silence proportion: fraction of samples whose absolute amplitude is
    below `silence_rel_threshold` * peak amplitude of this recording
    (a simple, interpretable, per-file-adaptive threshold — not a fixed
    dB value, since recordings are not level-normalized).
    """
    n_samples = len(y)
    duration = n_samples / sr if sr else 0.0
    peak_amplitude = float(np.max(np.abs(y))) if n_samples else 0.0
    rms = float(np.sqrt(np.mean(np.square(y)))) if n_samples else 0.0

    if n_samples and peak_amplitude > 0:
        threshold = silence_rel_threshold * peak_amplitude
        silence_proportion = float(np.mean(np.abs(y) < threshold))
    else:
        silence_proportion = 1.0

    return {
        "n_samples": n_samples,
        "sample_rate": sr,
        "duration_s": duration,
        "peak_amplitude": peak_amplitude,
        "rms": rms,
        "silence_proportion": silence_proportion,
    }
