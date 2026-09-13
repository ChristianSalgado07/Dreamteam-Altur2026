"""
Basic exploratory plots for a handful of representative recordings.

Not meant to be exhaustive — just enough to visually sanity-check that
Channel 0 extraction, F0 tracking, and the denoising path behave as
expected before trusting the bulk numbers in recording_level.csv.

Run with: python -m preprocessing.visualize
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

from preprocessing.audio_io import AUDIO_DIR, MANIFEST_CSV, load_channel0
from preprocessing.denoise import denoise_channel
from preprocessing.features import extract_all
from preprocessing.pipeline import has_turn_data, OUTPUT_DIR

PLOTS_DIR = OUTPUT_DIR / "plots"


def pick_representative_examples() -> list:
    """
    One example per (label, has_turn_data) combination, so the plots
    cover both classes and both turn-metadata situations.
    """
    with open(MANIFEST_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    wanted = [
        ("human", True), ("human", False),
        ("synthetic", True), ("synthetic", False),
    ]
    chosen = []
    for label, td in wanted:
        for row in rows:
            aid = row["anon_id"]
            if row["label"] == label and has_turn_data(aid) == td:
                chosen.append(aid)
                break
    return chosen


def plot_recording(anon_id: str):
    y, sr = load_channel0(anon_id)
    y_dn, _ = denoise_channel(y, sr)

    ts_orig, _ = extract_all(y, sr)
    ts_dn, _ = extract_all(y_dn, sr)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- spectrogram (original Channel 0) ---
    fig, ax = plt.subplots(figsize=(10, 4))
    S = librosa.stft(y, n_fft=1024, hop_length=256)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis="time", y_axis="hz", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"{anon_id} — Channel 0 spectrogram (original)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{anon_id}_spectrogram.png", dpi=120)
    plt.close(fig)

    # --- F0 trajectory: original vs denoised ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ts_orig["time"], ts_orig["f0"], label="original", alpha=0.8)
    ax.plot(ts_dn["time"], ts_dn["f0"], label="denoised", alpha=0.8, linestyle="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title(f"{anon_id} — F0 trajectory (unvoiced frames are gaps)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{anon_id}_f0.png", dpi=120)
    plt.close(fig)

    # --- RMS trajectory: original vs denoised ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ts_orig["time"], ts_orig["rms"], label="original", alpha=0.8)
    ax.plot(ts_dn["time"], ts_dn["rms"], label="denoised", alpha=0.8, linestyle="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("RMS")
    ax.set_title(f"{anon_id} — RMS energy trajectory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{anon_id}_rms.png", dpi=120)
    plt.close(fig)

    # --- spectral centroid trajectory: original vs denoised ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ts_orig["time"], ts_orig["spectral_centroid"], label="original", alpha=0.8)
    ax.plot(ts_dn["time"], ts_dn["spectral_centroid"], label="denoised", alpha=0.8, linestyle="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Spectral centroid (Hz)")
    ax.set_title(f"{anon_id} — spectral centroid trajectory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"{anon_id}_spectral_centroid.png", dpi=120)
    plt.close(fig)


def run():
    examples = pick_representative_examples()
    print(f"Representative examples: {examples}")
    for anon_id in examples:
        print(f"plotting {anon_id} ...")
        plot_recording(anon_id)
    print(f"Saved plots to {PLOTS_DIR}")
    return examples


if __name__ == "__main__":
    run()
