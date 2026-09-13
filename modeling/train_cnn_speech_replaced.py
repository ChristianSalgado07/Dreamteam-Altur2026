"""
CONTROLLED REPLACEMENT EXPERIMENT -- NOT a production change.

Follow-up to modeling/train_cnn_speech_only.py. That experiment REMOVED
near-silent windows from training, which incidentally shrank the total
training-window count for ~38% of recordings (those with fewer than 40
available speech windows) and coincided with a calibration collapse.
This experiment isolates the "silence content" variable more cleanly:
for every training recording, the TOTAL NUMBER of windows used stays
IDENTICAL to the baseline (same N per recording) -- near-silent windows
are REPLACED with additional (or reused) non-silent speech windows from
the SAME recording, never discarded outright and never borrowed from
another recording or the opposite class.

Does not modify modeling/train_cnn.py, modeling/train_cnn_speech_only.py,
app/decision.py, app/model.py, or any other production file. Does not
overwrite outputs/models/cnn_live_detector.pt or outputs/models/
cnn_speech_only.pt -- both are only ever read (loaded fresh for
evaluation), never retrained or written here.

Run with: python -m modeling.train_cnn_speech_replaced
"""
import json
import time
from collections import Counter
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
from preprocessing.features import extract_rms, extract_spectral, extract_f0

CNN_BASELINE_WEIGHTS_PATH = MODELS_DIR / "cnn_baseline.pt"          # from the previous experiment, read-only here
CNN_SPEECH_ONLY_WEIGHTS_PATH = MODELS_DIR / "cnn_speech_only.pt"    # from the previous experiment, read-only here
CNN_SPEECH_REPLACED_WEIGHTS_PATH = MODELS_DIR / "cnn_speech_replaced.pt"
COMPARISON_REPORT_MD = REPORTS_DIR / "cnn_speech_replaced_comparison.md"
COMPARISON_DUMP_JSON = REPORTS_DIR / "_speech_replaced_comparison_dump.json"
DIAGNOSTIC_DUMP_JSON = REPORTS_DIR / "_diagnostic_raw_dump.json"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "plots" / "cnn_speech_replaced"

dump = {}


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ----------------------------------------------------------------------
def select_replacement_indices(available: list, n_needed: int) -> list:
    """
    Deterministic (no RNG): if enough unique unused speech windows exist,
    picks n_needed of them evenly spread across `available` (temporal
    diversity, same even-spacing convention as subsample_windows
    elsewhere in this project). If not enough exist, cycles through
    `available` round-robin (still fully deterministic) -- reuse is
    tracked and reported, never silent.
    """
    if n_needed <= 0:
        return []
    if not available:
        return []  # caller handles the "no speech available at all" edge case
    if len(available) >= n_needed:
        idx = np.linspace(0, len(available) - 1, n_needed).round().astype(int)
        return [available[i] for i in idx]
    result = []
    i = 0
    while len(result) < n_needed:
        result.append(available[i % len(available)])
        i += 1
    return result


def build_speech_replaced_windows_for_recordings(anon_ids: list, labels_by_id: dict):
    """
    For each recording: reconstructs the baseline's exact selected window
    INDICES (same subsample_windows call, same quota), splits them into
    speech vs near-silent (same SILENCE_RMS_THRESHOLD as the live system
    and the prior experiments), and replaces each near-silent selection
    with a same-recording, not-already-selected speech window (or a
    deterministic reuse if the recording doesn't have enough unique
    speech windows). Total window count per recording is IDENTICAL to
    baseline's, by construction.
    """
    X, y, groups = [], [], []
    per_recording = []
    total_replaced = 0
    recordings_needing_replacement = 0
    recordings_with_reuse = 0
    recordings_with_no_speech_available = 0
    max_reuse_count = 0

    for anon_id in anon_ids:
        y_audio, sr = load_channel0(anon_id)
        raw_windows = make_windows(y_audio)      # unnormalized -- for the silence decision
        norm_windows = slice_windows(y_audio, sr)  # whole-recording normalized -- what actually gets log-mel'd
        assert len(raw_windows) == len(norm_windows)
        n_total = len(raw_windows)

        is_speech = [float(np.sqrt(np.mean(np.square(w)))) >= SILENCE_RMS_THRESHOLD for w in raw_windows]
        baseline_selected = subsample_windows(list(range(n_total)), MAX_WINDOWS_PER_RECORDING)

        speech_selected = [i for i in baseline_selected if is_speech[i]]
        silent_selected = [i for i in baseline_selected if not is_speech[i]]
        n_needed = len(silent_selected)

        all_speech_indices = [i for i in range(n_total) if is_speech[i]]
        already = set(speech_selected)
        available_replacements = [i for i in all_speech_indices if i not in already]

        if n_needed == 0:
            replacement_indices = []
        elif not available_replacements:
            # No speech windows left in this recording at all beyond what's
            # already selected -- cannot replace without violating
            # "same recording only". Documented exception: keep the
            # silent windows for this recording only.
            replacement_indices = silent_selected
            recordings_with_no_speech_available += 1
            per_recording.append({"anon_id": anon_id, "n_silent": n_needed, "replaced": 0,
                                   "reuse": 0, "note": "no unused speech windows available -- kept silent"})
        else:
            replacement_indices = select_replacement_indices(available_replacements, n_needed)
            counts = Counter(replacement_indices)
            reuse_count = sum(c - 1 for c in counts.values() if c > 1)
            max_reuse_this = max(counts.values()) if counts else 0
            if reuse_count > 0:
                recordings_with_reuse += 1
            total_replaced += n_needed
            recordings_needing_replacement += 1
            max_reuse_count = max(max_reuse_count, max_reuse_this)
            per_recording.append({"anon_id": anon_id, "n_silent": n_needed, "replaced": n_needed, "reuse": reuse_count})

        final_indices = speech_selected + replacement_indices
        assert len(final_indices) == len(baseline_selected), \
            f"{anon_id}: window count changed ({len(final_indices)} vs {len(baseline_selected)})"

        label = LABEL_TO_INT[labels_by_id[anon_id]]
        for i in final_indices:
            X.append(log_mel_spectrogram(norm_windows[i]))
            y.append(label)
            groups.append(anon_id)

    stats = {
        "total_replaced": total_replaced,
        "recordings_needing_replacement": recordings_needing_replacement,
        "recordings_with_reuse": recordings_with_reuse,
        "recordings_with_no_speech_available": recordings_with_no_speech_available,
        "max_reuse_count": int(max_reuse_count),
        "per_recording": per_recording,
    }
    return np.stack(X), np.array(y, dtype=np.float32), np.array(groups), stats


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
        return torch.sigmoid(model(torch.from_numpy(X).unsqueeze(1))).numpy()


def score_summary(scores: np.ndarray) -> dict:
    return {
        "n": int(len(scores)), "mean": float(np.mean(scores)), "median": float(np.median(scores)),
        "std": float(np.std(scores)), "min": float(np.min(scores)), "max": float(np.max(scores)),
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


def window_f0(raw_window, sr):
    _, f0, voiced_flag, _ = extract_f0(raw_window, sr)
    voiced = f0[~np.isnan(f0)]
    if len(voiced) < 8:
        return None
    return float(np.median(voiced))


def run():
    section("0. Setup + baseline pool reconciliation")
    if not CNN_BASELINE_WEIGHTS_PATH.exists() or not CNN_SPEECH_ONLY_WEIGHTS_PATH.exists():
        raise SystemExit("Run modeling/train_cnn_speech_only.py first -- its checkpoints are reused read-only here.")
    model_baseline = load_model(CNN_BASELINE_WEIGHTS_PATH)
    model_speech_only = load_model(CNN_SPEECH_ONLY_WEIGHTS_PATH)
    print("Loaded cnn_baseline.pt and cnn_speech_only.pt read-only (not retrained).")

    manifest = pd.read_csv(MANIFEST_CSV)
    labels_by_id = dict(zip(manifest["anon_id"], manifest["label"]))
    train_ids = manifest[manifest["split"] == "train"]["anon_id"].tolist()
    val_ids = manifest[manifest["split"] == "val"]["anon_id"].tolist()
    print(f"train={len(train_ids)} val={len(val_ids)} overlap={len(set(train_ids)&set(val_ids))} (must be 0)")
    assert len(set(train_ids) & set(val_ids)) == 0

    # --------------------------------------------------------------
    section("WINDOW SAMPLING RECONCILIATION (why 53.2% vs 6.2%)")
    # --------------------------------------------------------------
    print("""
  baseline pipeline (modeling/train_cnn.py):
    subsample_windows(slice_windows(y, sr), 40)
    -- evenly-spaced INDICES are picked from the FULL per-recording window
       pool (silence included), capped at 40/recording.

  diagnostic's 53.2% (outputs/reports/cnn_hallucination_diagnostics.md):
    measures what fraction of BASELINE's already-selected 40-per-recording
    windows are near-silent. Denominator = baseline's SELECTED windows.

  speech-only experiment's 6.2% removed (train_cnn_speech_only.py):
    filtered near-silent windows OUT of the full pool FIRST, then re-ran
    the same even-spacing 40-cap on the smaller (speech-only) remaining
    pool. Most recordings (measured: 176/282, 62.4%) still had >=40 speech
    windows available even after filtering, so they kept a full 40-window
    quota (just now 100% speech) -- total count barely dropped. Only the
    106/282 (37.6%) recordings with FEWER than 40 available speech windows
    contributed fewer total windows, which is the entire source of that 6.2%.

  CONCLUSION: 53.2% and 6.2% are not directly comparable -- they measure
  different denominators (selected-window composition vs. total selected
  count). This experiment fixes that ambiguity by holding the PER-RECORDING
  WINDOW COUNT fixed at exactly baseline's own count, and replacing (not
  discarding) near-silent selections with same-recording speech windows.
""")

    # --------------------------------------------------------------
    section("1. Build BASELINE training windows + SHARED validation windows (unchanged, reused)")
    # --------------------------------------------------------------
    t0 = time.time()
    X_train_base, y_train_base, g_train_base = build_windows_for_recordings(train_ids, labels_by_id)
    print(f"  baseline train windows: {len(y_train_base)} ({time.time()-t0:.0f}s)")
    t0 = time.time()
    X_val, y_val, g_val = build_windows_for_recordings(val_ids, labels_by_id)
    print(f"  SHARED validation windows (all 3 models): {len(y_val)} ({time.time()-t0:.0f}s)")

    # --------------------------------------------------------------
    section("2. Build SPEECH-REPLACED training windows")
    # --------------------------------------------------------------
    t0 = time.time()
    X_train_repl, y_train_repl, g_train_repl, repl_stats = build_speech_replaced_windows_for_recordings(train_ids, labels_by_id)
    print(f"  speech-replaced train windows: {len(y_train_repl)} ({time.time()-t0:.0f}s)")
    print(f"  windows replaced: {repl_stats['total_replaced']}")
    print(f"  recordings needing replacement: {repl_stats['recordings_needing_replacement']}/{len(train_ids)}")
    print(f"  recordings requiring reuse (not enough unique speech windows): {repl_stats['recordings_with_reuse']}")
    print(f"  max reuse count for any single window: {repl_stats['max_reuse_count']}")
    print(f"  recordings with NO speech windows available at all: {repl_stats['recordings_with_no_speech_available']}")

    # --------------------------------------------------------------
    section("3. Verification before training")
    # --------------------------------------------------------------
    def counts(y):
        return {"human": int((y == 0).sum()), "synthetic": int((y == 1).sum()), "total": int(len(y))}

    base_counts = counts(y_train_base)
    repl_counts = counts(y_train_repl)
    print("BASELINE TRAINING WINDOWS:", base_counts)
    print("SPEECH-REPLACED TRAINING WINDOWS:", repl_counts)
    print(f"Match (must be equal): {base_counts['total'] == repl_counts['total']}")

    per_rec_base = pd.Series(g_train_base).value_counts()
    per_rec_repl = pd.Series(g_train_repl).value_counts()
    print(f"windows/recording -- baseline: min={per_rec_base.min()} median={per_rec_base.median():.0f} max={per_rec_base.max()}")
    print(f"windows/recording -- replaced: min={per_rec_repl.min()} median={per_rec_repl.median():.0f} max={per_rec_repl.max()}")
    per_rec_identical = per_rec_base.sort_index().equals(per_rec_repl.sort_index())
    print(f"per-recording window counts IDENTICAL to baseline: {per_rec_identical}")
    assert per_rec_identical, "Per-recording window count changed -- STOPPING before training."
    assert set(g_train_base) == set(g_train_repl), "Recording ID set changed -- STOPPING before training."
    assert len(set(train_ids) & set(val_ids)) == 0, "Train/val leakage detected -- STOPPING before training."
    print("All pre-training verification checks PASSED.")

    dump["window_reconciliation"] = repl_stats
    dump["baseline_train_counts"] = base_counts
    dump["speech_replaced_train_counts"] = repl_counts

    # --------------------------------------------------------------
    section("4. Train speech_replaced_cnn (identical hyperparameters)")
    # --------------------------------------------------------------
    print(f"  epochs={EPOCHS} batch_size={BATCH_SIZE} lr={LR} seed={RANDOM_SEED}")
    model_repl = train_model(X_train_repl, y_train_repl)
    torch.save(model_repl.state_dict(), CNN_SPEECH_REPLACED_WEIGHTS_PATH)
    print(f"  saved -> {CNN_SPEECH_REPLACED_WEIGHTS_PATH}")

    models = {"baseline": model_baseline, "speech_only": model_speech_only, "speech_replaced": model_repl}

    # --------------------------------------------------------------
    section("5. Primary evaluation: SAME shared validation windows, all 3 models")
    # --------------------------------------------------------------
    primary_results = {}
    for name, model in models.items():
        probs = raw_scores(model, X_val)
        window_metrics = compute_metrics(y_val, probs)
        rec_scores = recording_level_scores(model, X_val, g_val)
        rec_ids = list(rec_scores.keys())
        rec_probs = np.array([rec_scores[i] for i in rec_ids])
        rec_labels = np.array([LABEL_TO_INT[labels_by_id[i]] for i in rec_ids])
        recording_metrics = compute_metrics(rec_labels, rec_probs)
        primary_results[name] = {"window_metrics": window_metrics, "recording_metrics": recording_metrics,
                                  "human_scores": score_summary(probs[y_val == 0]),
                                  "synthetic_scores": score_summary(probs[y_val == 1])}
        print(f"  [{name}] window ROC-AUC={window_metrics['roc_auc']:.4f}  "
              f"recording ROC-AUC={recording_metrics['roc_auc']:.4f}  "
              f"recording accuracy={recording_metrics['accuracy']:.4f}")

    # --------------------------------------------------------------
    section("6. Known human/synthetic recordings (raw scores)")
    # --------------------------------------------------------------
    known_human = [aid for aid in val_ids if labels_by_id[aid] == "human"][:3]
    known_synth = [aid for aid in val_ids if labels_by_id[aid] == "synthetic"][:3]
    known_examples = {}
    for name, model in models.items():
        known_examples[name] = {"human": {}, "synthetic": {}}
        print(f"  [{name}]")
        for aid in known_human:
            scores = recording_raw_scores(model, aid)
            known_examples[name]["human"][aid] = scores
            print(f"    HUMAN {aid}: {[f'{s:.3f}' for s in scores]}")
        for aid in known_synth:
            scores = recording_raw_scores(model, aid)
            known_examples[name]["synthetic"][aid] = scores
            print(f"    SYNTH {aid}: {[f'{s:.3f}' for s in scores]}")

    # --------------------------------------------------------------
    section("7. Synthetic stability check (all synthetic val windows)")
    # --------------------------------------------------------------
    synth_stability = {}
    for name, model in models.items():
        probs = raw_scores(model, X_val)
        s = score_summary(probs[y_val == 1])
        synth_stability[name] = s
        print(f"  [{name}] synthetic val windows: mean={s['mean']:.3f} median={s['median']:.3f} "
              f"std={s['std']:.3f} min={s['min']:.3f} max={s['max']:.3f} "
              f">0.65={s['pct_gt_065']:.1f}% >0.80={s['pct_gt_080']:.1f}% >0.90={s['pct_gt_090']:.1f}%")

    # --------------------------------------------------------------
    section("8. External evaluation (existing set, no new data)")
    # --------------------------------------------------------------
    external_results = {}
    if EXTERNAL_MANIFEST_CSV.exists() and EXTERNAL_PROCESSED_DIR.exists():
        ext_manifest = pd.read_csv(EXTERNAL_MANIFEST_CSV)
        paths_labels = [(EXTERNAL_PROCESSED_DIR / f"{row['external_id']}.wav", row["label"], row["external_id"])
                        for _, row in ext_manifest.iterrows()
                        if (EXTERNAL_PROCESSED_DIR / f"{row['external_id']}.wav").exists()]
        X_ext, y_ext, g_ext = build_windows_from_wavs(paths_labels)
        for name, model in models.items():
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
    section("9. Known false-positive windows -- all 3 models")
    # --------------------------------------------------------------
    fp_comparison = []
    if DIAGNOSTIC_DUMP_JSON.exists():
        diag = json.load(open(DIAGNOSTIC_DUMP_JSON))
        for w in diag["top10_suspicious_human_windows"]:
            aid, wi = w["anon_id"], w["window_index"]
            scores = {name: score_window_by_index(model, aid, wi) for name, model in models.items()}
            t = wi * 1.5
            row = {"anon_id": aid, "window_index": wi, "t_start": t, "t_end": t + 3.0,
                   "f0": w["f0_median"], "rms": w["rms"], "zcr": w["zcr"],
                   "spectral_centroid": w["spectral_centroid"], "spectral_bandwidth": w["spectral_bandwidth"],
                   "spectral_rolloff": w["spectral_rolloff"], **{f"{k}_score": v for k, v in scores.items()}}
            fp_comparison.append(row)
            f0str = f"{w['f0_median']:.0f}Hz" if w["f0_median"] else "n/a"
            print(f"  {aid} w{wi} [{t:5.1f}-{t+3:.1f}s] f0={f0str:>7}  "
                  f"baseline={scores['baseline']:.3f}  speech_only={scores['speech_only']:.3f}  "
                  f"speech_replaced={scores['speech_replaced']:.3f}")
    else:
        print("  diagnostic dump not found -- skipping")

    # --------------------------------------------------------------
    section("10. F0 shortcut check: score vs F0 across ALL human validation windows")
    # --------------------------------------------------------------
    human_val_idx = np.where(y_val == 0)[0]
    # Recompute the raw (pre-normalization) window + F0 for every human val window,
    # matched by group id + position -- reconstruct directly (cheap relative to pYIN cost either way).
    f0_by_window = []
    seen_per_recording = {}
    for aid in known_human:  # keep this section fast: known human recordings first (always included)
        pass
    # Build F0 list aligned with X_val/y_val ordering by recomputing per (anon_id) group segments.
    val_groups_list = list(g_val)
    idx_cursor = 0
    f0_values_full = [None] * len(y_val)
    for anon_id in val_ids:
        if labels_by_id[anon_id] != "human":
            continue
        y_audio, sr = load_channel0(anon_id)
        raw_all = make_windows(y_audio)
        norm_all = slice_windows(y_audio, sr)
        selected_idx = subsample_windows(list(range(len(raw_all))), MAX_WINDOWS_PER_RECORDING)
        # positions of this anon_id's windows within X_val/y_val/g_val (built in the same recording order)
        positions = [i for i, g in enumerate(val_groups_list) if g == anon_id]
        assert len(positions) == len(selected_idx)
        for pos, widx in zip(positions, selected_idx):
            f0_values_full[pos] = window_f0(raw_all[widx], sr)

    f0_arr = np.array([f0_values_full[i] if f0_values_full[i] is not None else np.nan for i in human_val_idx])
    band_mask = (f0_arr >= 125) & (f0_arr <= 155)
    f0_corr = {}
    for name, model in models.items():
        probs = raw_scores(model, X_val)[human_val_idx]
        valid = ~np.isnan(f0_arr)
        corr = float(np.corrcoef(f0_arr[valid], probs[valid])[0, 1]) if valid.sum() > 2 else None
        band_mean = float(np.mean(probs[band_mask & valid])) if (band_mask & valid).sum() > 0 else None
        outside_mean = float(np.mean(probs[~band_mask & valid])) if (~band_mask & valid).sum() > 0 else None
        f0_corr[name] = {"pearson_r_score_vs_f0": corr, "mean_score_125_155hz": band_mean,
                          "mean_score_outside_band": outside_mean, "n_in_band": int((band_mask & valid).sum()),
                          "n_valid_f0": int(valid.sum())}
        print(f"  [{name}] human windows with usable F0: {int(valid.sum())}/{len(f0_arr)} | "
              f"corr(score, F0)={corr} | mean score in 125-155Hz band={band_mean} "
              f"(n={int((band_mask & valid).sum())}) | mean score outside band={outside_mean}")

    # --------------------------------------------------------------
    section("11. Determinism check, all 3 models")
    # --------------------------------------------------------------
    fixed_mel = log_mel_spectrogram(slice_windows(*load_channel0(known_human[0]))[3])
    x_fixed = torch.from_numpy(fixed_mel).unsqueeze(0).unsqueeze(0)
    determinism = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            runs = [torch.sigmoid(model(x_fixed)).item() for _ in range(10)]
        determinism[name] = {"std": float(np.std(runs)), "all_equal": len(set(runs)) == 1}
        print(f"  [{name}] 10 runs, std={np.std(runs):.2e}, all_equal={len(set(runs))==1}")

    # --------------------------------------------------------------
    section("12. Score distribution plots")
    # --------------------------------------------------------------
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (name, model) in zip(axes, models.items()):
        probs = raw_scores(model, X_val)
        ax.hist(probs[y_val == 0], bins=30, alpha=0.6, label="human", color="#4caf50", range=(0, 1))
        ax.hist(probs[y_val == 1], bins=30, alpha=0.6, label="synthetic", color="#f44336", range=(0, 1))
        ax.axvline(0.65, color="#333", linestyle="--", linewidth=1)
        ax.axvline(0.35, color="#333", linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("raw CNN score")
        ax.legend()
    axes[0].set_ylabel("window count")
    plt.tight_layout()
    plot_path = PLOTS_DIR / "score_distribution_3way.png"
    plt.savefig(plot_path, dpi=110)
    plt.close()
    print(f"  saved -> {plot_path}")

    dump.update({
        "primary_results": primary_results, "known_examples": known_examples,
        "synth_stability": synth_stability, "external_results": external_results,
        "fp_comparison": fp_comparison, "f0_correlation": f0_corr, "determinism": determinism,
    })
    with open(COMPARISON_DUMP_JSON, "w") as f:
        json.dump(dump, f, indent=2, default=str)
    print(f"\nRaw comparison dump written to {COMPARISON_DUMP_JSON}")
    return dump


if __name__ == "__main__":
    run()
