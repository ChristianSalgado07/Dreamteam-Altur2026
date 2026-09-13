"""
Acoustic feature extraction for a single Channel-0 signal.

All time-aligned features (F0, RMS, spectral) are computed on the same
frame grid (FRAME_LENGTH / HOP_LENGTH below) so they can be combined
into one time-series table per recording.

Frame settings: 128 ms analysis window, 32 ms hop (~31 frames/sec).
At 8000 Hz this is long enough for pYIN to see several pitch periods
even for a low male F0 (~65 Hz => 15.4 ms/period), while still fine
enough to trace prosodic-level pitch/energy movement, which is what
this exploratory stage needs (not phoneme-level precision).
"""
import numpy as np
import librosa

FRAME_LENGTH = 1024
HOP_LENGTH = 256

F0_MIN_HZ = 50.0   # below typical human speech F0
F0_MAX_HZ = 500.0  # above typical human speech F0


def extract_f0(y: np.ndarray, sr: int):
    """
    pYIN pitch tracking. Unvoiced/unreliable frames come back as NaN
    (librosa's `fill_na=np.nan`) rather than 0 or another placeholder,
    so they are never silently treated as "pitch zero" in summary
    stats or plots. `voiced_flag` records which frames pYIN judged
    voiced at all, independent of the NaN filling.
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=F0_MIN_HZ,
        fmax=F0_MAX_HZ,
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        fill_na=np.nan,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH)
    return times, f0, voiced_flag, voiced_prob


def extract_rms(y: np.ndarray):
    rms = librosa.feature.rms(
        y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    return rms


def extract_spectral(y: np.ndarray, sr: int):
    centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    return centroid, bandwidth, rolloff, zcr


def extract_spectral_flatness(y: np.ndarray) -> np.ndarray:
    """
    Spectral flatness (Wiener entropy) per frame, same frame grid as the
    other spectral features above: near 1.0 = noise-like/flat spectrum,
    near 0.0 = tonal/peaked spectrum. Added for the ElevenLabs-fingerprint
    stage (see external_data/adapt_and_extract_elevenlabs.py) — not wired
    into extract_all()/the original recording_level.csv so the
    already-completed pipeline output doesn't change.
    """
    return librosa.feature.spectral_flatness(
        y=y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]


N_MFCC = 13


def extract_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Standard MFCCs (librosa), on the same frame grid as the other
    features above. Returns an (N_MFCC, n_frames) array — added for the
    modeling-preparation stage (see modeling/prepare_dataset.py); not
    wired into extract_all()/the existing recording_level.csv or
    timeseries.csv so the already-completed acoustic pipeline output
    doesn't need to be regenerated.
    """
    return librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH
    )


def mfcc_summary(mfcc: np.ndarray) -> dict:
    """
    Per-coefficient mean/std across time — 26 columns (13 mean + 13 std),
    deliberately not a larger feature space (no per-quantile columns;
    "do not create an unnecessarily huge feature space").
    """
    summary = {}
    for i in range(mfcc.shape[0]):
        coef = mfcc[i]
        summary[f"mfcc{i + 1}_mean"] = float(np.mean(coef))
        summary[f"mfcc{i + 1}_std"] = float(np.std(coef))
    return summary


def nan_summary(values, prefix: str) -> dict:
    """
    Mean/std/min/max/range over the non-NaN values only. If every
    value is NaN (e.g. a recording pYIN finds entirely unvoiced),
    the summary is NaN rather than a fabricated number.
    """
    values = np.asarray(values, dtype=np.float64)
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return {
            f"mean_{prefix}": np.nan,
            f"std_{prefix}": np.nan,
            f"min_{prefix}": np.nan,
            f"max_{prefix}": np.nan,
            f"range_{prefix}": np.nan,
        }
    return {
        f"mean_{prefix}": float(np.mean(valid)),
        f"std_{prefix}": float(np.std(valid)),
        f"min_{prefix}": float(np.min(valid)),
        f"max_{prefix}": float(np.max(valid)),
        f"range_{prefix}": float(np.max(valid) - np.min(valid)),
    }


def extract_all(y: np.ndarray, sr: int) -> dict:
    """
    Run every extractor on one signal and return both the time-series
    arrays (for the per-recording time-series output) and the
    recording-level summary dict.
    """
    times, f0, voiced_flag, voiced_prob = extract_f0(y, sr)
    rms = extract_rms(y)
    centroid, bandwidth, rolloff, zcr = extract_spectral(y, sr)

    n = min(len(times), len(rms), len(centroid), len(bandwidth), len(rolloff), len(zcr))

    timeseries = {
        "time": times[:n],
        "f0": f0[:n],
        "voiced_flag": voiced_flag[:n],
        "rms": rms[:n],
        "spectral_centroid": centroid[:n],
        "spectral_bandwidth": bandwidth[:n],
        "spectral_rolloff": rolloff[:n],
        "zcr": zcr[:n],
    }

    voiced_proportion = float(np.mean(voiced_flag)) if len(voiced_flag) else np.nan

    summary = {}
    summary.update(nan_summary(f0[:n], "f0"))
    summary["voiced_proportion"] = voiced_proportion
    summary.update(nan_summary(rms[:n], "rms"))
    summary.update(nan_summary(centroid[:n], "spectral_centroid"))
    summary.update(nan_summary(bandwidth[:n], "spectral_bandwidth"))
    summary.update(nan_summary(rolloff[:n], "spectral_rolloff"))
    summary.update(nan_summary(zcr[:n], "zcr"))

    return timeseries, summary
