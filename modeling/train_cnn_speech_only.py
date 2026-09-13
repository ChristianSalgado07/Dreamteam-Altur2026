"""
CONTROLLED A/B EXPERIMENT -- NOT a replacement for the production model.

Tests the primary hypothesis from outputs/reports/cnn_hallucination_diagnostics.md:
does excluding near-silent windows from TRAINING (not live inference, which already
filters them) reduce the CNN's false-positive behavior on real human speech?

Everything is held constant except the training window content:
  - Same VoiceCNN architecture (app/model.py, imported unchanged)
  - Same recording-level train/val split (files/csv/manifest.csv, unchanged)
  - Same optimizer/loss/LR/batch size/epochs/random seed (imported from
    modeling/train_cnn.py, not re-typed, so they cannot drift)
  - Same VALIDATION windows for both models (this script does not touch
    the val-window construction at all -- reuses
    modeling.train_cnn.build_windows_for_recordings unchanged, exactly
    once, and evaluates BOTH models against that one shared result)
  - Same external evaluation set (external_data/, no new data)

The ONLY changed variable: speech_only_cnn's TRAINING windows exclude any
window whose RAW (pre-normalization) RMS is below
app.decision.SILENCE_RMS_THRESHOLD -- the EXACT existing live silence
threshold, not a new one invented for this experiment.

Does not modify modeling/train_cnn.py, app/decision.py, app/model.py, or
any other production file. Does not overwrite the production checkpoint
(outputs/models/cnn_live_detector.pt) -- that file is only ever read
(copied to cnn_baseline.pt for a clean side-by-side name), never written.

Run with: python -m modeling.train_cnn_speech_only
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from preprocessing.audio_io import load_channel0, MANIFEST_CSV
from app.audio_utils import make_windows, slice_windows, log_mel_spectrogram
from app.model import VoiceCNN, load_model
from app.decision import SILENCE_RMS_THRESHOLD
from modeling.train_cnn import (
    subsample_windows, build_windows_for_recordings, build_windows_from_wavs,
    WindowDataset, recording_level_scores, compute_metrics,
    LABEL_TO_INT, MAX_WINDOWS_PER_RECORDING, RANDOM_SEED, BATCH_SIZE, EPOCHS, LR,
    EXTERNAL_MANIFEST_CSV, EXTERNAL_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR,
)

PRODUCTION_CHECKPOINT = MODELS_DIR / "cnn_live_detector.pt"
CNN_BASELINE_WEIGHTS_PATH = MODELS_DIR / "cnn_baseline.pt"
CNN_SPEECH_ONLY_WEIGHTS_PATH = MODELS_DIR / "cnn_speech_only.pt"
COMPARISON_REPORT_MD = REPORTS_DIR / "cnn_speech_only_comparison.md"
COMPARISON_DUMP_JSON = REPORTS_DIR / "_speech_only_comparison_dump.json"
DIAGNOSTIC_DUMP_JSON = REPORTS_DIR / "_diagnostic_raw_dump.json"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "plots" / "cnn_speech_only"

dump = {}


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ----------------------------------------------------------------------
def build_speech_only_windows_for_recordings(anon_ids: list, labels_by_id: dict) -> tuple:
    """
    Identical to modeling.train_cnn.build_windows_for_recordings EXCEPT
    windows whose RAW (pre-normalization) RMS falls below the EXISTING
    live silence threshold (app.decision.SILENCE_RMS_THRESHOLD) are
    dropped before the even-spacing subsample step. Reuses every other
    existing function unchanged: load_channel0, make_windows,
    slice_windows, subsample_windows, log_mel_spectrogram.
    """
    X, y, groups = [], [], []
    per_recording_counts = []
    for anon_id in anon_ids:
        y_audio, sr = load_channel0(anon_id)
        raw_windows = make_windows(y_audio)            # UNNORMALIZED -- for the silence decision only
        norm_windows = slice_windows(y_audio, sr)        # whole-recording normalized -- same as baseline training
        assert len(raw_windows) == len(norm_windows)

        speech_idx = [i for i, w in enumerate(raw_windows)
                      if float(np.sqrt(np.mean(np.square(w)))) >= SILENCE_RMS_THRESHOLD]
        speech_norm_windows = [norm_windows[i] for i in speech_idx]
        selected = subsample_windows(speech_norm_windows, MAX_WINDOWS_PER_RECORDING)

        label = LABEL_TO_INT[labels_by_id[anon_id]]
        for w in selected:
            X.append(log_mel_spectrogram(w))
            y.append(label)
            groups.append(anon_id)
        per_recording_counts.append({
            "anon_id": anon_id, "label": labels_by_id[anon_id],
            "total_windows": len(raw_windows), "speech_windows": len(speech_idx),
            "selected_windows": len(selected),
        })
    return np.stack(X), np.array(y, dtype=np.float32), np.array(groups), per_recording_counts


def train_model(X_train, y_train, seed=RANDOM_SEED) -> VoiceCNN:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = VoiceCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(WindowDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        print(f"    epoch {epoch:2d}/{EPOCHS}  loss={total_loss/len(y_train):.4f}")
    model.eval()
    return model


def raw_scores(model, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).unsqueeze(1))
        return torch.sigmoid(logits).numpy()


def score_summary(scores: np.ndarray) -> dict:
    return {
        "n": int(len(scores)), "mean": float(np.mean(scores)), "median": float(np.median(scores)),
        "min": float(np.min(scores)), "max": float(np.max(scores)),
        "pct_gt_065": float(np.mean(scores > 0.65) * 100),
        "pct_gt_080": float(np.mean(scores > 0.80) * 100),
        "pct_gt_090": float(np.mean(scores > 0.90) * 100),
    }


def recording_raw_scores(model, anon_id, n=6):
    y_audio, sr = load_channel0(anon_id)
    windows = slice_windows(y_audio, sr)
    idx = np.linspace(0, len(windows) - 1, min(n, len(windows))).round().astype(int)
    scores = []
    model.eval()
    with torch.no_grad():
        for i in idx:
            mel = log_mel_spectrogram(windows[i])
            x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
            scores.append(torch.sigmoid(model(x)).item())
    return scores


def score_window_by_index(model, anon_id, window_index):
    y_audio, sr = load_channel0(anon_id)
    windows = slice_windows(y_audio, sr)
    mel = log_mel_spectrogram(windows[window_index])
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return torch.sigmoid(model(x)).item()


def run():
    section("0. Setup: load manifest, copy baseline checkpoint (never retrained/overwritten)")
    if not PRODUCTION_CHECKPOINT.exists():
        raise SystemExit(f"Production checkpoint not found at {PRODUCTION_CHECKPOINT} -- nothing to compare against.")
    shutil.copy(PRODUCTION_CHECKPOINT, CNN_BASELINE_WEIGHTS_PATH)
    print(f"Copied {PRODUCTION_CHECKPOINT.name} -> {CNN_BASELINE_WEIGHTS_PATH.name} (production file untouched)")

    manifest = pd.read_csv(MANIFEST_CSV)
    labels_by_id = dict(zip(manifest["anon_id"], manifest["label"]))
    train_ids = manifest[manifest["split"] == "train"]["anon_id"].tolist()
    val_ids = manifest[manifest["split"] == "val"]["anon_id"].tolist()
    print(f"train={len(train_ids)} val={len(val_ids)} (existing split, unchanged)")

    # --------------------------------------------------------------
    section("1. Build BASELINE training windows (existing, unfiltered function) + SHARED validation windows")
    # --------------------------------------------------------------
    t0 = time.time()
    X_train_base, y_train_base, g_train_base = build_windows_for_recordings(train_ids, labels_by_id)
    print(f"  baseline train windows: {len(y_train_base)} ({time.time()-t0:.0f}s)")
    t0 = time.time()
    X_val, y_val, g_val = build_windows_for_recordings(val_ids, labels_by_id)
    print(f"  SHARED validation windows (used for BOTH models): {len(y_val)} ({time.time()-t0:.0f}s)")

    # --------------------------------------------------------------
    section("2. Build SPEECH-ONLY training windows (near-silent windows excluded)")
    # --------------------------------------------------------------
    t0 = time.time()
    X_train_speech, y_train_speech, g_train_speech, per_rec = build_speech_only_windows_for_recordings(train_ids, labels_by_id)
    print(f"  speech-only train windows: {len(y_train_speech)} ({time.time()-t0:.0f}s)")

    # --------------------------------------------------------------
    section("3. Filtering effect report")
    # --------------------------------------------------------------
    def counts(y):
        return {"human": int((y == 0).sum()), "synthetic": int((y == 1).sum()), "total": int(len(y))}

    base_counts = counts(y_train_base)
    speech_counts = counts(y_train_speech)
    removed = {k: base_counts[k] - speech_counts[k] for k in base_counts}
    pct_removed = {k: (100 * removed[k] / base_counts[k] if base_counts[k] else 0.0) for k in base_counts}

    print("BASELINE TRAINING WINDOWS")
    print(f"  Human: {base_counts['human']}  Synthetic: {base_counts['synthetic']}  Total: {base_counts['total']}")
    print("SPEECH-ONLY TRAINING WINDOWS")
    print(f"  Human: {speech_counts['human']}  Synthetic: {speech_counts['synthetic']}  Total: {speech_counts['total']}")
    print("REMOVED")
    print(f"  Human: {removed['human']} ({pct_removed['human']:.1f}%)  "
          f"Synthetic: {removed['synthetic']} ({pct_removed['synthetic']:.1f}%)  "
          f"Total: {removed['total']} ({pct_removed['total']:.1f}%)")

    base_human_pct = 100 * base_counts["human"] / base_counts["total"]
    speech_human_pct = 100 * speech_counts["human"] / speech_counts["total"]
    print(f"\nClass balance -- baseline: human={base_human_pct:.1f}% synthetic={100-base_human_pct:.1f}%")
    print(f"Class balance -- speech-only: human={speech_human_pct:.1f}% synthetic={100-speech_human_pct:.1f}%")
    print("(train_cnn.py uses no class weighting/balancing mechanism -- none is added here either, preserved as-is)")

    dump["filtering_effect"] = {
        "baseline_counts": base_counts, "speech_only_counts": speech_counts,
        "removed": removed, "pct_removed": pct_removed,
        "baseline_human_pct": base_human_pct, "speech_only_human_pct": speech_human_pct,
    }

    # --------------------------------------------------------------
    section("4. Train speech_only_cnn (identical procedure/hyperparameters to modeling/train_cnn.py)")
    # --------------------------------------------------------------
    print(f"  epochs={EPOCHS} batch_size={BATCH_SIZE} lr={LR} seed={RANDOM_SEED}")
    model_speech = train_model(X_train_speech, y_train_speech)
    torch.save(model_speech.state_dict(), CNN_SPEECH_ONLY_WEIGHTS_PATH)
    print(f"  saved -> {CNN_SPEECH_ONLY_WEIGHTS_PATH}")

    model_baseline = load_model(CNN_BASELINE_WEIGHTS_PATH)  # loaded fresh, never retrained here

    # --------------------------------------------------------------
    section("5A. Primary evaluation: SAME shared validation windows, both models")
    # --------------------------------------------------------------
    results = {}
    for name, model in [("baseline", model_baseline), ("speech_only", model_speech)]:
        probs = raw_scores(model, X_val)
        window_metrics = compute_metrics(y_val, probs)
        rec_scores = recording_level_scores(model, X_val, g_val)
        rec_ids = list(rec_scores.keys())
        rec_probs = np.array([rec_scores[i] for i in rec_ids])
        rec_labels = np.array([LABEL_TO_INT[labels_by_id[i]] for i in rec_ids])
        recording_metrics = compute_metrics(rec_labels, rec_probs)
        results[name] = {"window_metrics": window_metrics, "recording_metrics": recording_metrics}
        print(f"  [{name}] window ROC-AUC={window_metrics['roc_auc']:.4f}  "
              f"recording ROC-AUC={recording_metrics['roc_auc']:.4f}  "
              f"recording accuracy={recording_metrics['accuracy']:.4f}")

    # --------------------------------------------------------------
    section("5B. Known human / synthetic recordings (raw scores, no decision layer)")
    # --------------------------------------------------------------
    known_human = [aid for aid in val_ids if labels_by_id[aid] == "human"][:3]
    known_synth = [aid for aid in val_ids if labels_by_id[aid] == "synthetic"][:3]
    known_examples = {}
    for name, model in [("baseline", model_baseline), ("speech_only", model_speech)]:
        known_examples[name] = {"human": {}, "synthetic": {}}
        print(f"  [{name}] HUMAN:")
        for aid in known_human:
            scores = recording_raw_scores(model, aid)
            known_examples[name]["human"][aid] = score_summary(np.array(scores))
            print(f"    {aid}: {[f'{s:.3f}' for s in scores]}")
        print(f"  [{name}] SYNTHETIC:")
        for aid in known_synth:
            scores = recording_raw_scores(model, aid)
            known_examples[name]["synthetic"][aid] = score_summary(np.array(scores))
            print(f"    {aid}: {[f'{s:.3f}' for s in scores]}")

    # --------------------------------------------------------------
    section("5C. External evaluation (existing OpenSLR + gTTS/espeak set, no new data)")
    # --------------------------------------------------------------
    external_results = {}
    if EXTERNAL_MANIFEST_CSV.exists() and EXTERNAL_PROCESSED_DIR.exists():
        ext_manifest = pd.read_csv(EXTERNAL_MANIFEST_CSV)
        paths_labels = [(EXTERNAL_PROCESSED_DIR / f"{row['external_id']}.wav", row["label"], row["external_id"])
                        for _, row in ext_manifest.iterrows()
                        if (EXTERNAL_PROCESSED_DIR / f"{row['external_id']}.wav").exists()]
        X_ext, y_ext, g_ext = build_windows_from_wavs(paths_labels)
        for name, model in [("baseline", model_baseline), ("speech_only", model_speech)]:
            ext_rec_scores = recording_level_scores(model, X_ext, g_ext)
            ext_ids = list(ext_rec_scores.keys())
            ext_probs = np.array([ext_rec_scores[i] for i in ext_ids])
            ext_id_to_label = dict(zip(ext_manifest["external_id"], ext_manifest["label"]))
            ext_labels = np.array([LABEL_TO_INT[ext_id_to_label[i]] for i in ext_ids])
            m = compute_metrics(ext_labels, ext_probs)
            external_results[name] = m
            print(f"  [{name}] external recording ROC-AUC={m['roc_auc']:.4f} accuracy={m['accuracy']:.4f}")
    else:
        print("  external set not found -- skipping")

    # --------------------------------------------------------------
    section("6. Human false-positive comparison (the diagnostic's exact suspicious windows)")
    # --------------------------------------------------------------
    fp_comparison = []
    if DIAGNOSTIC_DUMP_JSON.exists():
        diag = json.load(open(DIAGNOSTIC_DUMP_JSON))
        for w in diag["top10_suspicious_human_windows"]:
            aid, wi = w["anon_id"], w["window_index"]
            b_score = score_window_by_index(model_baseline, aid, wi)
            s_score = score_window_by_index(model_speech, aid, wi)
            t = wi * 1.5
            row = {
                "anon_id": aid, "window_index": wi, "t_start": t, "t_end": t + 3.0,
                "f0": w["f0_median"], "rms": w["rms"], "zcr": w["zcr"],
                "spectral_centroid": w["spectral_centroid"], "spectral_bandwidth": w["spectral_bandwidth"],
                "spectral_rolloff": w["spectral_rolloff"],
                "baseline_score": b_score, "speech_only_score": s_score,
            }
            fp_comparison.append(row)
            f0str = f"{w['f0_median']:.0f}Hz" if w["f0_median"] else "n/a"
            print(f"  {aid} w{wi} [{t:5.1f}-{t+3:.1f}s] f0={f0str:>7}  "
                  f"baseline={b_score:.3f}  speech_only={s_score:.3f}  Δ={s_score-b_score:+.3f}")
    else:
        print("  diagnostic dump not found -- skipping")

    dump["known_examples"] = known_examples
    dump["primary_val_results"] = results
    dump["external_results"] = external_results
    dump["fp_comparison"] = fp_comparison

    # --------------------------------------------------------------
    section("7. F0 shortcut check: is the 130-150Hz pattern still present in speech_only?")
    # --------------------------------------------------------------
    fp_rows_with_f0 = [r for r in fp_comparison if r["f0"] is not None]
    if fp_rows_with_f0:
        base_mean = np.mean([r["baseline_score"] for r in fp_rows_with_f0])
        speech_mean = np.mean([r["speech_only_score"] for r in fp_rows_with_f0])
        print(f"  {len(fp_rows_with_f0)} of the suspicious windows have a measured F0 in ~130-150Hz.")
        print(f"  Mean score on these windows -- baseline: {base_mean:.3f}  speech_only: {speech_mean:.3f}")
        dump["f0_shortcut_check"] = {"n_windows": len(fp_rows_with_f0), "baseline_mean": float(base_mean), "speech_only_mean": float(speech_mean)}

    # --------------------------------------------------------------
    section("8. Determinism check, both models, same tensor, 10 runs")
    # --------------------------------------------------------------
    fixed_mel = log_mel_spectrogram(slice_windows(*load_channel0(known_human[0]))[3])
    x_fixed = torch.from_numpy(fixed_mel).unsqueeze(0).unsqueeze(0)
    det = {}
    for name, model in [("baseline", model_baseline), ("speech_only", model_speech)]:
        model.eval()
        with torch.no_grad():
            runs = [torch.sigmoid(model(x_fixed)).item() for _ in range(10)]
        det[name] = {"runs": runs, "std": float(np.std(runs))}
        print(f"  [{name}] 10 runs, std={np.std(runs):.2e}  all_equal={len(set(runs))==1}")
    dump["determinism"] = det

    # --------------------------------------------------------------
    section("9. Score distribution plots")
    # --------------------------------------------------------------
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    human_mask = (y_val == 0)
    synth_mask = (y_val == 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (name, model) in zip(axes, [("baseline", model_baseline), ("speech_only", model_speech)]):
        probs = raw_scores(model, X_val)
        ax.hist(probs[human_mask], bins=30, alpha=0.6, label="human", color="#4caf50", range=(0, 1))
        ax.hist(probs[synth_mask], bins=30, alpha=0.6, label="synthetic", color="#f44336", range=(0, 1))
        ax.axvline(0.65, color="#333", linestyle="--", linewidth=1)
        ax.axvline(0.35, color="#333", linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("raw CNN score (AI-likelihood)")
        ax.legend()
    axes[0].set_ylabel("window count")
    plt.tight_layout()
    plot_path = PLOTS_DIR / "score_distribution_comparison.png"
    plt.savefig(plot_path, dpi=110)
    plt.close()
    print(f"  saved -> {plot_path}")

    with open(COMPARISON_DUMP_JSON, "w") as f:
        json.dump(dump, f, indent=2, default=str)
    print(f"\nRaw comparison dump written to {COMPARISON_DUMP_JSON}")

    return dump, results, known_examples, external_results, fp_comparison, det, per_rec


if __name__ == "__main__":
    run()
