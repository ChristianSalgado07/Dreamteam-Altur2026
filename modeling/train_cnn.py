"""
Trains the small log-Mel CNN live-detector prototype (see app/model.py,
app/audio_utils.py). Additive: does not touch, retrain, or replace the
existing LR/RF/SVM models (modeling/baseline.py,
outputs/models/locked_*.joblib) or any prior report.

Data: files/csv/manifest.csv (353 recordings, existing train/val split --
NOT re-randomized) + files/audio/*.wav, Channel 0 only (Channel 1 is the
known-AI-agent channel per rules.md and is never used as a training
target here).

Leakage rule: split happens at the RECORDING level BEFORE windowing.
train recordings -> train windows, val recordings -> val windows.
Windows from one recording never appear on both sides.

Windows are capped at MAX_WINDOWS_PER_RECORDING (evenly spaced across
the recording) so long recordings don't dominate the window count and
training stays fast on CPU -- a hackathon-scope pragmatic choice, not a
new confound (still label-blind, applied identically to every
recording).

External evaluation reuses the ALREADY-BUILT external validation set
(external_data/processed/original/*.wav + external_manifest.csv, 40
OpenSLR human + 48 gTTS/espeak synthetic clips -- see
external_data/README.md) run through the exact same app.audio_utils
preprocessing. No new external data collection here. ElevenLabs holdout
is NOT included -- generation was blocked on safe credential handling
this session; this is reported explicitly rather than silently omitted.

Run with: python -m modeling.train_cnn
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from preprocessing.audio_io import load_channel0, MANIFEST_CSV, AUDIO_DIR
from app.audio_utils import slice_windows, log_mel_spectrogram, PREPROCESSING_CONFIG
from app.model import VoiceCNN

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "outputs" / "models"
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"
CNN_WEIGHTS_PATH = MODELS_DIR / "cnn_live_detector.pt"
CNN_CONFIG_PATH = MODELS_DIR / "cnn_preprocessing_config.json"
CNN_REPORT_MD = REPORTS_DIR / "cnn_training_report.md"

EXTERNAL_DIR = REPO_ROOT / "external_data"
EXTERNAL_MANIFEST_CSV = EXTERNAL_DIR / "manifests" / "external_manifest.csv"
EXTERNAL_PROCESSED_DIR = EXTERNAL_DIR / "processed" / "original"

MAX_WINDOWS_PER_RECORDING = 40
LABEL_TO_INT = {"human": 0, "synthetic": 1}  # 1 = synthetic/AI, matches modeling/baseline.py convention

RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 12
LR = 1e-3


def subsample_windows(windows: list, max_n: int) -> list:
    if len(windows) <= max_n:
        return windows
    idx = np.linspace(0, len(windows) - 1, max_n).round().astype(int)
    return [windows[i] for i in idx]


def build_windows_for_recordings(anon_ids: list, labels_by_id: dict) -> tuple:
    """Returns (X, y, group_ids) -- one row per window."""
    X, y, groups = [], [], []
    for anon_id in anon_ids:
        y_audio, sr = load_channel0(anon_id)
        raw_windows = subsample_windows(slice_windows(y_audio, sr), MAX_WINDOWS_PER_RECORDING)
        label = LABEL_TO_INT[labels_by_id[anon_id]]
        for w in raw_windows:
            X.append(log_mel_spectrogram(w))
            y.append(label)
            groups.append(anon_id)
    return np.stack(X), np.array(y, dtype=np.float32), np.array(groups)


def build_windows_from_wavs(paths_labels: list) -> tuple:
    """Same as above but for arbitrary (path, label, id) wav files not
    under files/audio (used for the external evaluation set)."""
    X, y, groups = [], [], []
    for path, label, ext_id in paths_labels:
        y_audio, sr = sf.read(str(path), always_2d=True)
        y_audio = y_audio[:, 0]
        raw_windows = subsample_windows(slice_windows(y_audio, sr), MAX_WINDOWS_PER_RECORDING)
        for w in raw_windows:
            X.append(log_mel_spectrogram(w))
            y.append(LABEL_TO_INT[label])
            groups.append(ext_id)
    if not X:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))
    return np.stack(X), np.array(y, dtype=np.float32), np.array(groups)


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).unsqueeze(1)  # (N, 1, n_mels, n_frames)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def recording_level_scores(model, X, groups) -> dict:
    """Mean predicted probability per recording/group id."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).unsqueeze(1))
        probs = torch.sigmoid(logits).numpy()
    df = pd.DataFrame({"group": groups, "prob": probs})
    return df.groupby("group")["prob"].mean().to_dict()


def compute_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


def print_metrics(title: str, metrics: dict):
    print(f"\n=== {title} (n={metrics['n']}) ===")
    print(f"  accuracy : {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall   : {metrics['recall']:.4f}")
    print(f"  f1       : {metrics['f1']:.4f}")
    print(f"  roc_auc  : {metrics['roc_auc']:.4f}")
    print(f"  confusion matrix [[TN,FP],[FN,TP]]: {metrics['confusion_matrix']}")
    if metrics["roc_auc"] >= 0.98:
        print("  FLAG: suspiciously perfect performance -- see report caveats "
              "(the same pattern the hand-crafted-feature models showed; "
              "treat as descriptive of this dataset, not a generalization claim).")


def run():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    manifest = pd.read_csv(MANIFEST_CSV)
    labels_by_id = dict(zip(manifest["anon_id"], manifest["label"]))
    train_ids = manifest[manifest["split"] == "train"]["anon_id"].tolist()
    val_ids = manifest[manifest["split"] == "val"]["anon_id"].tolist()
    print(f"Recordings: {len(train_ids)} train, {len(val_ids)} val (existing split, unchanged)")

    t0 = time.time()
    print("Building training windows (Channel 0 only)...")
    X_train, y_train, g_train = build_windows_for_recordings(train_ids, labels_by_id)
    print(f"  {X_train.shape[0]} train windows from {len(train_ids)} recordings ({time.time()-t0:.0f}s)")

    t0 = time.time()
    print("Building validation windows...")
    X_val, y_val, g_val = build_windows_for_recordings(val_ids, labels_by_id)
    print(f"  {X_val.shape[0]} val windows from {len(val_ids)} recordings ({time.time()-t0:.0f}s)")

    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    model = VoiceCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    print(f"\nTraining CNN for {EPOCHS} epochs...")
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        print(f"  epoch {epoch:2d}/{EPOCHS}  loss={total_loss/len(y_train):.4f}")

    # --- Window-level validation metrics ---
    model.eval()
    with torch.no_grad():
        val_logits = model(torch.from_numpy(X_val).unsqueeze(1))
        val_probs = torch.sigmoid(val_logits).numpy()
    window_metrics = compute_metrics(y_val, val_probs)
    print_metrics("INTERNAL VAL -- window level", window_metrics)

    # --- Recording-level validation metrics (mean window prob per recording) ---
    rec_scores = recording_level_scores(model, X_val, g_val)
    rec_ids = list(rec_scores.keys())
    rec_probs = np.array([rec_scores[i] for i in rec_ids])
    rec_labels = np.array([LABEL_TO_INT[labels_by_id[i]] for i in rec_ids])
    recording_metrics = compute_metrics(rec_labels, rec_probs)
    print_metrics("INTERNAL VAL -- recording level (mean of window probs)", recording_metrics)

    # --- External validation (existing OpenSLR + gTTS + espeak set) ---
    external_metrics = None
    if EXTERNAL_MANIFEST_CSV.exists() and EXTERNAL_PROCESSED_DIR.exists():
        ext_manifest = pd.read_csv(EXTERNAL_MANIFEST_CSV)
        paths_labels = []
        for _, row in ext_manifest.iterrows():
            wav_path = EXTERNAL_PROCESSED_DIR / f"{row['external_id']}.wav"
            if wav_path.exists():
                paths_labels.append((wav_path, row["label"], row["external_id"]))
        X_ext, y_ext, g_ext = build_windows_from_wavs(paths_labels)
        if len(y_ext):
            ext_rec_scores = recording_level_scores(model, X_ext, g_ext)
            ext_ids = list(ext_rec_scores.keys())
            ext_probs = np.array([ext_rec_scores[i] for i in ext_ids])
            ext_id_to_label = dict(zip(ext_manifest["external_id"], ext_manifest["label"]))
            ext_labels = np.array([LABEL_TO_INT[ext_id_to_label[i]] for i in ext_ids])
            external_metrics = compute_metrics(ext_labels, ext_probs)
            print_metrics("EXTERNAL (OpenSLR human + gTTS/espeak synthetic) -- recording level", external_metrics)
    else:
        print("\nExternal validation set not found -- skipping (see external_data/README.md).")

    print("\nElevenLabs holdout: NOT AVAILABLE this session -- generation was blocked on "
          "safe API-credential handling (see external_data/generate_elevenlabs.py). "
          "Not fabricated; reported honestly as missing.")

    # --- Save model + preprocessing config ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CNN_WEIGHTS_PATH)
    config = dict(PREPROCESSING_CONFIG)
    config.update({
        "max_windows_per_recording": MAX_WINDOWS_PER_RECORDING,
        "label_convention": LABEL_TO_INT,
        "train_recordings": len(train_ids),
        "val_recordings": len(val_ids),
        "train_windows": int(X_train.shape[0]),
        "val_windows": int(X_val.shape[0]),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "random_seed": RANDOM_SEED,
    })
    with open(CNN_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved model -> {CNN_WEIGHTS_PATH}")
    print(f"Saved preprocessing config -> {CNN_CONFIG_PATH}")

    # --- Report ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CNN live-detector training report\n",
        "Additive prototype: a small log-Mel-spectrogram CNN trained on ",
        "Channel 0 of the existing dataset, for real-time microphone ",
        "inference (see app/main.py, app/live_infer.py). Does not replace ",
        "or retrain the existing LR/RF/SVM models "
        "(see modeling/README.md) -- a separate model family on a "
        "different (spectrogram, not hand-crafted-feature) representation.\n",
        f"\n## Data\n",
        f"- Train: {len(train_ids)} recordings -> {X_train.shape[0]} windows "
        f"(existing split from files/csv/manifest.csv, unchanged)\n",
        f"- Val: {len(val_ids)} recordings -> {X_val.shape[0]} windows\n",
        f"- Window: {PREPROCESSING_CONFIG['window_seconds']}s, "
        f"{int(PREPROCESSING_CONFIG['overlap']*100)}% overlap, "
        f"capped at {MAX_WINDOWS_PER_RECORDING} windows/recording (evenly spaced)\n",
        f"- Channel 1 (known AI agent) was never used as a training target, per rules.md.\n",
        "\n## Results\n",
        "### Internal validation -- window level\n",
        f"```\n{json.dumps(window_metrics, indent=2)}\n```\n",
        "### Internal validation -- recording level (mean window prob)\n",
        f"```\n{json.dumps(recording_metrics, indent=2)}\n```\n",
    ]
    if external_metrics is not None:
        lines.append("### External (OpenSLR human + gTTS/espeak synthetic) -- recording level\n")
        lines.append(f"```\n{json.dumps(external_metrics, indent=2)}\n```\n")
    else:
        lines.append("### External validation\nNot available (set missing at run time).\n")
    lines.append(
        "\n### ElevenLabs holdout\n"
        "Not available this session -- ElevenLabs generation was blocked on "
        "safe API-credential handling (see external_data/generate_elevenlabs.py "
        "and external_data/README.md). Not fabricated.\n"
    )
    if window_metrics["roc_auc"] >= 0.98 or recording_metrics["roc_auc"] >= 0.98:
        lines.append(
            "\n## Caveat: suspiciously high internal performance\n"
            "Internal ROC-AUC is at/near ceiling, echoing the same pattern "
            "already documented for the hand-crafted-feature models "
            "(outputs/reports/modeling_evaluation_report.md, "
            "external_validation_report.md, domain_matched_validation_report.md): "
            "near-perfect internal separation on this dataset has NOT reliably "
            "transferred to external speech. Treat internal numbers as a "
            "description of this dataset, not a working general detector, "
            "until external/held-out numbers (above) corroborate it.\n"
        )
    with open(CNN_REPORT_MD, "w") as f:
        f.writelines(lines)
    print(f"Saved report -> {CNN_REPORT_MD}")


if __name__ == "__main__":
    run()
