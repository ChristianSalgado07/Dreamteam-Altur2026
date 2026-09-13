"""
======================================================================
DIAGNOSTIC-ONLY SCRIPT -- NOT PRODUCTION CODE.

Does NOT train, retrain, or modify any model, checkpoint, threshold,
preprocessing step, or app/ file. Loads outputs/models/cnn_baseline.pt
and outputs/models/cnn_speech_replaced.pt read-only. Safe to delete
after outputs/reports/cnn_silence_diagnostic.md is read.
======================================================================

Follow-up to outputs/reports/cnn_hallucination_diagnostics.md and
cnn_speech_replaced_comparison.md: both found ~0.99 raw CNN scores on
near-total/literal digital silence, unaffected by speech-replaced
training. This script isolates exactly what the CNN does on silence.

Pipeline used for EVERY diagnostic window, matching live inference
(app/live_infer.py:RollingDetector.step()) exactly, EXCEPT the silence
gate is deliberately bypassed (never done in production/live use) so we
can measure what the raw CNN outputs underneath it:

  raw 3s window -> normalize_rms (PER WINDOW, live-style, not the
  whole-recording-then-slice style training uses) -> log_mel_spectrogram
  -> CNN -> sigmoid -> raw score

No smoothing, no hysteresis, no decision engine, no silence cooldown.

Categories are built PURELY from acoustic measurements (RMS/peak) on the
RAW, pre-normalization window -- never from the human/synthetic label.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from preprocessing.audio_io import load_channel0, MANIFEST_CSV
from preprocessing.features import extract_rms, extract_spectral, extract_f0
from preprocessing.normalize import normalize_rms
from app.audio_utils import make_windows, TARGET_SR, WINDOW_SECONDS, log_mel_spectrogram
from app.model import load_model
from app.decision import SILENCE_RMS_THRESHOLD

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "outputs" / "models"
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"
PLOTS_DIR = REPO_ROOT / "outputs" / "plots" / "cnn_silence_diagnostic"
DUMP_PATH = REPORTS_DIR / "_silence_diagnostic_dump.json"
DIAGNOSTIC_DUMP_JSON = REPORTS_DIR / "_diagnostic_raw_dump.json"

DIGITAL_EPS = 1e-6          # peak amplitude below this -> "digital silence"
CLEAR_SPEECH_RMS_MIN = 0.03  # comfortably above the live gate (0.01) -> "clear speech"
SUBCHUNK_SECONDS = 0.5

dump = {}


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


# ----------------------------------------------------------------------
def categorize_window(raw_window: np.ndarray, sr: int) -> str:
    """
    Purely acoustic (RMS/peak), never label-based. Returns one of:
    'digital_silence', 'near_silence', 'speech_adjacent', 'clear_speech'.
    """
    rms = float(np.sqrt(np.mean(np.square(raw_window))))
    peak = float(np.max(np.abs(raw_window)))
    if peak < DIGITAL_EPS:
        return "digital_silence"

    n_sub = int(SUBCHUNK_SECONDS * sr)
    sub_rms = [float(np.sqrt(np.mean(np.square(raw_window[i:i + n_sub]))))
               for i in range(0, len(raw_window), n_sub) if len(raw_window[i:i + n_sub]) == n_sub]
    has_burst = any(s >= SILENCE_RMS_THRESHOLD for s in sub_rms)
    has_quiet = any(s < SILENCE_RMS_THRESHOLD * 0.5 for s in sub_rms)

    if rms < SILENCE_RMS_THRESHOLD:
        return "speech_adjacent" if has_burst else "near_silence"
    if rms >= CLEAR_SPEECH_RMS_MIN:
        return "clear_speech"
    # borderline zone (0.01 <= rms < 0.03): speech-adjacent if genuinely mixed, else count as speech
    return "speech_adjacent" if (has_burst and has_quiet) else "clear_speech"


def live_style_score(model, raw_window: np.ndarray):
    """EXACT live path minus the silence gate: per-window normalize_rms -> log-mel -> CNN -> sigmoid."""
    y_norm, norm_info = normalize_rms(raw_window)
    mel = log_mel_spectrogram(y_norm)
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        score = torch.sigmoid(model(x)).item()
    return score, mel, norm_info


def window_features(raw_window, sr):
    rms_frames = extract_rms(raw_window)
    centroid, bandwidth, rolloff, zcr = extract_spectral(raw_window, sr)
    _, f0, voiced_flag, _ = extract_f0(raw_window, sr)
    voiced = f0[~np.isnan(f0)]
    return {
        "rms": float(np.sqrt(np.mean(np.square(raw_window)))),
        "peak": float(np.max(np.abs(raw_window))),
        "zcr": float(np.mean(zcr)),
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
        "spectral_rolloff": float(np.mean(rolloff)),
        "f0": float(np.median(voiced)) if len(voiced) >= 8 else None,
    }


def score_summary(scores: np.ndarray) -> dict:
    if len(scores) == 0:
        return {"n": 0}
    return {
        "n": int(len(scores)), "mean": float(np.mean(scores)), "median": float(np.median(scores)),
        "min": float(np.min(scores)), "max": float(np.max(scores)),
        "pct_gt_065": float(np.mean(scores > 0.65) * 100),
        "pct_gt_080": float(np.mean(scores > 0.80) * 100),
        "pct_gt_090": float(np.mean(scores > 0.90) * 100),
    }


def run():
    section("1-2. Load both models, build the recording sample (human + synthetic, no label-based selection of windows)")
    baseline = load_model(MODELS_DIR / "cnn_baseline.pt")
    speech_replaced = load_model(MODELS_DIR / "cnn_speech_replaced.pt")
    models = {"baseline": baseline, "speech_replaced": speech_replaced}
    print("Loaded cnn_baseline.pt and cnn_speech_replaced.pt (read-only, no retraining).")

    manifest = pd.read_csv(MANIFEST_CSV)
    human_ids = manifest[manifest["label"] == "human"]["anon_id"].tolist()
    synth_ids = manifest[manifest["label"] == "synthetic"]["anon_id"].tolist()
    sample_ids = sorted(human_ids)[:15] + sorted(synth_ids)[:15]  # deterministic, label used only to ensure BOTH classes represented in the SAMPLE, not to build categories
    print(f"Sampling {len(sample_ids)} recordings (15 human + 15 synthetic) to build diagnostic windows from.")

    # --------------------------------------------------------------
    section("3. Build the controlled window set -- pure acoustic categorization")
    # --------------------------------------------------------------
    all_windows = []  # list of dicts: anon_id, label, window_index, t_start, category, raw_window (kept only transiently)
    for anon_id in sample_ids:
        y_audio, sr = load_channel0(anon_id)
        raw_windows = make_windows(y_audio)
        label = manifest.set_index("anon_id").loc[anon_id, "label"]
        for i, w in enumerate(raw_windows):
            cat = categorize_window(w, sr)
            all_windows.append({"anon_id": anon_id, "label": label, "window_index": i,
                                 "t_start": i * (WINDOW_SECONDS * 0.5), "category": cat, "window": w})

    cat_counts = pd.Series([w["category"] for w in all_windows]).value_counts()
    print("Category counts across all scanned windows:")
    print(cat_counts)
    cat_counts_by_label = pd.DataFrame(all_windows).groupby(["category", "label"]).size().unstack(fill_value=0)
    print("\nCategory counts by label (label was NOT used to build categories -- shown only to check both classes appear in each):")
    print(cat_counts_by_label)
    dump["category_counts"] = cat_counts.to_dict()
    dump["category_counts_by_label"] = cat_counts_by_label.to_dict()

    # Cap per category (for the expensive F0 step) but score ALL windows with both CNNs (cheap)
    MAX_PER_CATEGORY_FOR_FEATURES = 60
    by_category = {}
    for cat in ["digital_silence", "near_silence", "speech_adjacent", "clear_speech"]:
        items = [w for w in all_windows if w["category"] == cat]
        by_category[cat] = items

    # --------------------------------------------------------------
    section("4-7. Score every category window with BOTH models (raw CNN, no decision layer)")
    # --------------------------------------------------------------
    category_results = {}
    detailed_rows = []
    for cat, items in by_category.items():
        scores = {"baseline": [], "speech_replaced": []}
        feature_sample = items[:MAX_PER_CATEGORY_FOR_FEATURES] if len(items) > MAX_PER_CATEGORY_FOR_FEATURES else items
        # deterministic even-spread sample if we have more than the cap
        if len(items) > MAX_PER_CATEGORY_FOR_FEATURES:
            idx = np.linspace(0, len(items) - 1, MAX_PER_CATEGORY_FOR_FEATURES).round().astype(int)
            feature_sample = [items[i] for i in idx]

        for item in items:
            for name, model in models.items():
                s, _, _ = live_style_score(model, item["window"])
                scores[name].append(s)

        for item in feature_sample:
            feats = window_features(item["window"], TARGET_SR)
            b_score, _, _ = live_style_score(baseline, item["window"])
            r_score, _, _ = live_style_score(speech_replaced, item["window"])
            detailed_rows.append({"anon_id": item["anon_id"], "label": item["label"],
                                   "window_index": item["window_index"], "category": cat,
                                   "baseline_score": b_score, "speech_replaced_score": r_score, **feats})

        category_results[cat] = {name: score_summary(np.array(v)) for name, v in scores.items()}
        for name in models:
            s = category_results[cat][name]
            if s["n"]:
                print(f"  [{cat:16s}][{name:16s}] n={s['n']:4d} mean={s['mean']:.3f} median={s['median']:.3f} "
                      f"min={s['min']:.3f} max={s['max']:.3f} >0.65={s['pct_gt_065']:.1f}% "
                      f">0.80={s['pct_gt_080']:.1f}% >0.90={s['pct_gt_090']:.1f}%")

    dump["category_results"] = category_results
    dump["detailed_rows"] = detailed_rows

    # --------------------------------------------------------------
    section("8. Pure digital-silence arrays (synthetic test signals, never used for training)")
    # --------------------------------------------------------------
    window_samples = int(WINDOW_SECONDS * TARGET_SR)
    pure_zero_3s = np.zeros(window_samples, dtype=np.float64)
    pure_zero_1s_padded = np.concatenate([np.zeros(int(1.0 * TARGET_SR)), np.zeros(window_samples - int(1.0 * TARGET_SR))])
    rng = np.random.RandomState(42)
    very_low_noise = rng.randn(window_samples) * 1e-4
    extremely_low_noise = rng.randn(window_samples) * 1e-6

    synthetic_tests = {
        "pure_zero_3.0s": pure_zero_3s,
        "pure_zero_1.0s_padded_to_3s": pure_zero_1s_padded,
        "very_low_gaussian_noise_1e-4": very_low_noise,
        "extremely_low_gaussian_noise_1e-6": extremely_low_noise,
    }
    synthetic_results = {}
    for name, sig in synthetic_tests.items():
        row = {"rms": float(np.sqrt(np.mean(sig ** 2))), "peak": float(np.max(np.abs(sig)))}
        for mname, model in models.items():
            score, mel, norm_info = live_style_score(model, sig)
            row[f"{mname}_score"] = score
            row[f"{mname}_gain"] = norm_info["gain"]
            row[f"{mname}_degenerate"] = norm_info["degenerate_signal"]
            row[f"{mname}_logmel_min"] = float(mel.min())
            row[f"{mname}_logmel_max"] = float(mel.max())
            row[f"{mname}_logmel_mean"] = float(mel.mean())
            row[f"{mname}_logmel_std"] = float(mel.std())
        synthetic_results[name] = row
        print(f"  {name}: rms={row['rms']:.2e} peak={row['peak']:.2e} "
              f"baseline_score={row['baseline_score']:.4f} speech_replaced_score={row['speech_replaced_score']:.4f} "
              f"(gain={row['baseline_gain']:.2f}, degenerate={row['baseline_degenerate']})")
    dump["synthetic_test_results"] = synthetic_results

    # --------------------------------------------------------------
    section("9. Tensor statistics before the CNN: silence vs near-silence vs speech")
    # --------------------------------------------------------------
    tensor_stats = {}
    representative = {}
    for cat in ["digital_silence", "near_silence", "speech_adjacent", "clear_speech"]:
        items = by_category.get(cat, [])
        if items:
            representative[cat] = items[len(items) // 2]["window"]
    for cat, w in representative.items():
        y_norm, norm_info = normalize_rms(w)
        mel = log_mel_spectrogram(y_norm)
        tensor_stats[cat] = {
            "raw_min": float(w.min()), "raw_max": float(w.max()),
            "raw_rms": float(np.sqrt(np.mean(w ** 2))),
            "post_normalize_rms": norm_info["current_rms"], "gain_applied": norm_info["gain"],
            "logmel_min": float(mel.min()), "logmel_max": float(mel.max()),
            "logmel_mean": float(mel.mean()), "logmel_std": float(mel.std()),
        }
        print(f"  [{cat}] raw_rms={tensor_stats[cat]['raw_rms']:.5f} gain={tensor_stats[cat]['gain_applied']:.2f} "
              f"logmel: min={tensor_stats[cat]['logmel_min']:.1f} max={tensor_stats[cat]['logmel_max']:.1f} "
              f"mean={tensor_stats[cat]['logmel_mean']:.1f} std={tensor_stats[cat]['logmel_std']:.1f}")
    dump["tensor_stats"] = tensor_stats

    # --------------------------------------------------------------
    section("10. Baseline vs speech-replaced agreement across all diagnostic windows")
    # --------------------------------------------------------------
    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed["delta"] = df_detailed["speech_replaced_score"] - df_detailed["baseline_score"]
    print("Largest INCREASES (speech_replaced higher than baseline):")
    print(df_detailed.sort_values("delta", ascending=False)[["anon_id", "window_index", "category", "baseline_score", "speech_replaced_score", "delta"]].head(10).to_string(index=False))
    print("\nLargest DECREASES (speech_replaced lower than baseline):")
    print(df_detailed.sort_values("delta")[["anon_id", "window_index", "category", "baseline_score", "speech_replaced_score", "delta"]].head(10).to_string(index=False))
    print("\nMean |delta| by category:")
    print(df_detailed.groupby("category")["delta"].agg(["mean", "std", "count"]))
    dump["agreement_top_increases"] = df_detailed.sort_values("delta", ascending=False).head(10).to_dict("records")
    dump["agreement_top_decreases"] = df_detailed.sort_values("delta").head(10).to_dict("records")

    # --------------------------------------------------------------
    section("11. Revisit the exact 3 known silence false positives + neighbors")
    # --------------------------------------------------------------
    three_silence_fp = []
    if DIAGNOSTIC_DUMP_JSON.exists():
        diag = json.load(open(DIAGNOSTIC_DUMP_JSON))
        candidates = [w for w in diag["top10_suspicious_human_windows"]
                      if w["anon_id"] in ("call_c4fc8a865292", "call_bd98a8224da0")]
        for w in candidates:
            three_silence_fp.append((w["anon_id"], w["window_index"]))
    print(f"Target windows: {three_silence_fp}")

    neighbor_timelines = {}
    for anon_id, wi in three_silence_fp:
        y_audio, sr = load_channel0(anon_id)
        raw_windows = make_windows(y_audio)
        lo, hi = max(0, wi - 3), min(len(raw_windows), wi + 4)
        rows = []
        for i in range(lo, hi):
            w = raw_windows[i]
            rms = float(np.sqrt(np.mean(w ** 2)))
            b_score, _, _ = live_style_score(baseline, w)
            r_score, _, _ = live_style_score(speech_replaced, w)
            rows.append({"window_index": i, "t_start": i * 1.5, "rms": rms,
                         "baseline_score": b_score, "speech_replaced_score": r_score, "is_target": i == wi})
        neighbor_timelines[f"{anon_id}_w{wi}"] = rows
        print(f"\n  {anon_id} around window {wi}:")
        for r in rows:
            marker = "  <-- TARGET" if r["is_target"] else ""
            print(f"    t={r['t_start']:6.1f}s  rms={r['rms']:.5f}  baseline={r['baseline_score']:.3f}  "
                  f"replaced={r['speech_replaced_score']:.3f}{marker}")
    dump["silence_fp_neighbors"] = neighbor_timelines

    # --------------------------------------------------------------
    section("12. call_0e1e2f29bfdc regression -- identify high-scoring windows + neighbors")
    # --------------------------------------------------------------
    target_id = "call_0e1e2f29bfdc"
    y_audio, sr = load_channel0(target_id)
    raw_windows = make_windows(y_audio)
    idx6 = np.linspace(0, len(raw_windows) - 1, 6).round().astype(int)
    regression_rows = []
    for i in idx6:
        w = raw_windows[i]
        b_score, _, _ = live_style_score(baseline, w)
        r_score, _, _ = live_style_score(speech_replaced, w)
        regression_rows.append({"window_index": int(i), "t_start": i * 1.5,
                                 "baseline_score": b_score, "speech_replaced_score": r_score})
    print(f"{target_id} (the 6 evenly-spaced sampled windows used in prior reports):")
    for r in regression_rows:
        print(f"  idx={r['window_index']:3d} t={r['t_start']:6.1f}s baseline={r['baseline_score']:.3f} replaced={r['speech_replaced_score']:.3f}")

    high_scoring = [r for r in regression_rows if r["speech_replaced_score"] >= 0.5]
    print(f"\nHigh-scoring under speech_replaced (>=0.5): {len(high_scoring)}")
    regression_detail = []
    for r in high_scoring:
        wi = r["window_index"]
        lo, hi = max(0, wi - 3), min(len(raw_windows), wi + 4)
        neighbors = []
        for i in range(lo, hi):
            w = raw_windows[i]
            feats = window_features(w, sr)
            b_score, _, _ = live_style_score(baseline, w)
            r_score, _, _ = live_style_score(speech_replaced, w)
            cat = categorize_window(w, sr)
            neighbors.append({"window_index": i, "t_start": i * 1.5, "category": cat,
                               "baseline_score": b_score, "speech_replaced_score": r_score,
                               "is_target": i == wi, **feats})
        regression_detail.append({"target_window": wi, "neighbors": neighbors})
        print(f"\n  Around window {wi} (t={wi*1.5:.1f}s):")
        for n in neighbors:
            marker = "  <-- TARGET" if n["is_target"] else ""
            f0str = f"{n['f0']:.0f}Hz" if n["f0"] else "n/a"
            print(f"    idx={n['window_index']:3d} cat={n['category']:16s} rms={n['rms']:.4f} "
                  f"zcr={n['zcr']:.3f} centroid={n['spectral_centroid']:.0f}Hz f0={f0str:>7} "
                  f"baseline={n['baseline_score']:.3f} replaced={n['speech_replaced_score']:.3f}{marker}")
    dump["regression_detail"] = regression_detail
    dump["regression_6window_scores"] = regression_rows

    # --------------------------------------------------------------
    section("13-14. Score vs RMS / Score vs F0 plots")
    # --------------------------------------------------------------
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(detailed_rows)
    cat_colors = {"digital_silence": "#000000", "near_silence": "#666666",
                  "speech_adjacent": "#ffc107", "clear_speech": "#4caf50"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, model_col, title in zip(axes, ["baseline_score", "speech_replaced_score"], ["baseline", "speech_replaced"]):
        for cat, color in cat_colors.items():
            sub = df[df["category"] == cat]
            marker = "o" if sub["label"].eq("human").any() else "o"
            ax.scatter(sub[sub.label == "human"]["rms"], sub[sub.label == "human"][model_col],
                       color=color, marker="o", alpha=0.6, label=f"{cat} (human)", s=18)
            ax.scatter(sub[sub.label == "synthetic"]["rms"], sub[sub.label == "synthetic"][model_col],
                       color=color, marker="^", alpha=0.6, label=f"{cat} (synthetic)", s=18)
        ax.set_xscale("symlog", linthresh=1e-4)
        ax.axhline(0.65, color="#999", linestyle="--", linewidth=1)
        ax.axvline(SILENCE_RMS_THRESHOLD, color="#999", linestyle=":", linewidth=1)
        ax.set_xlabel("RMS (symlog scale)")
        ax.set_ylabel("raw CNN AI score")
        ax.set_title(title)
    axes[0].legend(fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5)) if False else None
    plt.tight_layout()
    rms_plot_path = PLOTS_DIR / "score_vs_rms.png"
    plt.savefig(rms_plot_path, dpi=110)
    plt.close()
    print(f"saved -> {rms_plot_path}")

    df_f0 = df[df["f0"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, model_col, title in zip(axes, ["baseline_score", "speech_replaced_score"], ["baseline", "speech_replaced"]):
        ax.scatter(df_f0[df_f0.label == "human"]["f0"], df_f0[df_f0.label == "human"][model_col],
                   color="#4caf50", alpha=0.6, label="human", s=18)
        ax.scatter(df_f0[df_f0.label == "synthetic"]["f0"], df_f0[df_f0.label == "synthetic"][model_col],
                   color="#f44336", alpha=0.6, label="synthetic", s=18, marker="^")
        ax.axvspan(125, 155, color="#ab47bc", alpha=0.15, label="125-155Hz")
        ax.axhline(0.65, color="#999", linestyle="--", linewidth=1)
        ax.set_xlabel("F0 (Hz)")
        ax.set_ylabel("raw CNN AI score")
        ax.set_title(title)
        ax.legend(fontsize=8)
    plt.tight_layout()
    f0_plot_path = PLOTS_DIR / "score_vs_f0.png"
    plt.savefig(f0_plot_path, dpi=110)
    plt.close()
    print(f"saved -> {f0_plot_path}")

    with open(DUMP_PATH, "w") as f:
        json.dump(dump, f, indent=2, default=str)
    print(f"\nRaw dump written to {DUMP_PATH}")
    return dump


if __name__ == "__main__":
    run()
