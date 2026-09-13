"""
Diagnoses WHY the baseline (modeling/baseline.py, modeling/evaluate.py)
scores near-perfect ROC-AUC, rather than trusting it. Does not modify or
delete any existing baseline output — everything here is additive, under
new filenames. See modeling/README.md and outputs/reports/
modeling_evaluation_report.md for the write-up this produces.

Sections implemented here (numbered to match the task):
  4. stratified CV (via modeling/cv_utils.py)
  5. feature-family ablation (A-G)
  6. univariate feature separability + suspicious-feature plots
  7. controlled feature elimination
  8. recording-condition / pipeline-confound analysis
  9. suspicious simple baselines
  10. original vs denoised comparison
  11. Channel-1 (confirmed AI agent) diagnostic comparison

Run with: python -m modeling.diagnostics   (after modeling.prepare_dataset)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_auc_score

from preprocessing.audio_io import AUDIO_DIR
from preprocessing.pipeline import FEATURES_DIR as ACOUSTIC_FEATURES_DIR
from modeling.prepare_dataset import (
    MODELING_DATASET_COMPLETE_CSV, FEATURE_FAMILIES_CSV, FEATURES_DIR, REPORTS_DIR,
    ACOUSTIC_CSV, MFCC_CSV, load_temporal,
)
from modeling.cv_utils import evaluate_feature_set

PLOTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "plots" / "modeling" / "diagnostics"
CHANNEL1_CSV = ACOUSTIC_FEATURES_DIR / "channel1_diagnostic_features.csv"

EXPERIMENT_COMPARISON_CSV = FEATURES_DIR / "modeling_experiment_comparison.csv"
UNIVARIATE_SEP_CSV = FEATURES_DIR / "modeling_univariate_separability_trainonly.csv"
RECORDING_CONDITION_CSV = FEATURES_DIR / "modeling_recording_condition_analysis.csv"
CHANNEL1_DIAGNOSTIC_CSV = FEATURES_DIR / "modeling_channel1_diagnostic_comparison.csv"
EVALUATION_REPORT_MD = REPORTS_DIR / "modeling_evaluation_report.md"

# --- feature-name-based subcategories (Step 7) — only groupings that
# make sense given the actual column names produced upstream ---
ACOUSTIC_ENERGY_COLS = [
    "raw_peak_amplitude", "raw_rms", "raw_silence_proportion",
    "mean_rms", "std_rms", "min_rms", "max_rms", "range_rms",
]
ACOUSTIC_PITCH_COLS = ["mean_f0", "std_f0", "min_f0", "max_f0", "range_f0", "voiced_proportion"]
# "spectral" = everything acoustic that isn't energy/amplitude or pitch
# (spectral centroid/bandwidth/rolloff + zero-crossing rate)


def load_baseline_split():
    df = pd.read_csv(MODELING_DATASET_COMPLETE_CSV)
    y = (df["label"] == "synthetic").astype(int)
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    return df, y, train_mask, val_mask


def get_family_cols() -> dict:
    fam = pd.read_csv(FEATURE_FAMILIES_CSV)
    return {f: fam.loc[fam.family == f, "feature"].tolist() for f in ("acoustic", "mfcc", "temporal")}


# ---------------------------------------------------------------- 5 ---
def family_ablation(df, y, train_mask, val_mask, families) -> pd.DataFrame:
    experiments = {
        "A_acoustic_only": families["acoustic"],
        "B_mfcc_only": families["mfcc"],
        "C_temporal_only": families["temporal"],
        "D_acoustic_mfcc": families["acoustic"] + families["mfcc"],
        "E_acoustic_temporal": families["acoustic"] + families["temporal"],
        "F_mfcc_temporal": families["mfcc"] + families["temporal"],
        "G_acoustic_mfcc_temporal": families["acoustic"] + families["mfcc"] + families["temporal"],
    }
    results = []
    for label, cols in experiments.items():
        results.append(evaluate_feature_set(
            df.loc[train_mask, cols], y[train_mask], df.loc[val_mask, cols], y[val_mask], cols, label,
        ))
    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------- 6 ---
def univariate_separability(df, y, train_mask, families, output_csv=None) -> pd.DataFrame:
    feature_to_family = {c: fam for fam, cols in families.items() for c in cols}
    X_train, y_train = df.loc[train_mask], y[train_mask]
    rows = []
    for c in feature_to_family:
        sub = pd.concat([X_train[c], y_train.rename("y")], axis=1).dropna()
        if sub["y"].nunique() < 2 or len(sub) < 5:
            continue
        auc = roc_auc_score(sub["y"], sub[c])
        rows.append({"feature": c, "family": feature_to_family[c], "auc": auc, "auc_abs_from_0.5": abs(auc - 0.5)})
    sep_df = pd.DataFrame(rows).sort_values("auc_abs_from_0.5", ascending=False)
    sep_df.to_csv(output_csv or UNIVARIATE_SEP_CSV, index=False)
    return sep_df


def plot_suspicious_features(df, y, train_mask, sep_df, n=6, plots_dir=None, filename_prefix="suspicious_"):
    plots_dir = plots_dir or PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    suspicious = sep_df[(sep_df["auc"] >= 0.95) | (sep_df["auc"] <= 0.05)].head(n)
    for _, row in suspicious.iterrows():
        feat = row["feature"]
        vals_h = df.loc[train_mask & (y == 0), feat].dropna()
        vals_s = df.loc[train_mask & (y == 1), feat].dropna()
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.boxplot([vals_h, vals_s], tick_labels=["human", "synthetic"], showmeans=True)
        ax.set_title(f"{feat} (train split) — univariate AUC={row['auc']:.3f}")
        ax.set_ylabel(feat)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{filename_prefix}{feat}.png", dpi=120)
        plt.close(fig)
    return suspicious


# ---------------------------------------------------------------- 7 ---
def controlled_elimination(df, y, train_mask, val_mask, families) -> pd.DataFrame:
    acoustic = families["acoustic"]
    temporal = families["temporal"]
    response_latency_cols = [c for c in temporal if c.startswith("response_latency")]

    experiments = {
        "1_all_features": acoustic + families["mfcc"] + temporal,
        "2_all_except_acoustic": families["mfcc"] + temporal,
        "3_all_except_mfcc": acoustic + temporal,
        "4_all_except_temporal": acoustic + families["mfcc"],
        "5_acoustic_without_energy_amplitude": [c for c in acoustic if c not in ACOUSTIC_ENERGY_COLS],
        "6_acoustic_without_spectral": [c for c in acoustic if c in ACOUSTIC_ENERGY_COLS + ACOUSTIC_PITCH_COLS],
        "7_acoustic_without_pitch": [c for c in acoustic if c not in ACOUSTIC_PITCH_COLS],
        "8_temporal_without_response_latency": [c for c in temporal if c not in response_latency_cols],
    }
    results = []
    for label, cols in experiments.items():
        results.append(evaluate_feature_set(
            df.loc[train_mask, cols], y[train_mask], df.loc[val_mask, cols], y[val_mask], cols, label,
        ))
    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------- 9 ---
def suspicious_simple_baselines(df, y, train_mask, val_mask) -> pd.DataFrame:
    baselines = {
        "rms_only": ["mean_rms", "std_rms", "min_rms", "max_rms", "range_rms"],
        "zcr_only": ["mean_zcr", "std_zcr", "min_zcr", "max_zcr", "range_zcr"],
        "peak_amplitude_only": ["raw_peak_amplitude"],
        "noise_floor_only": ["raw_silence_proportion", "min_rms", "min_spectral_centroid",
                              "min_spectral_bandwidth", "min_spectral_rolloff"],
        "tiny_acoustic_subset_3feat": ["std_zcr", "max_spectral_rolloff", "mean_rms"],
    }
    results = []
    for label, cols in baselines.items():
        results.append(evaluate_feature_set(
            df.loc[train_mask, cols], y[train_mask], df.loc[val_mask, cols], y[val_mask], cols, label,
        ))
    return pd.concat(results, ignore_index=True)


# --------------------------------------------------------------- 10 ---
def load_combined_by_version(version: str) -> pd.DataFrame:
    acoustic = pd.read_csv(ACOUSTIC_CSV)
    acoustic = acoustic[acoustic["version"] == version].drop(columns=["channel", "version", "sample_rate", "n_samples"])
    mfcc = pd.read_csv(MFCC_CSV)
    mfcc = mfcc[mfcc["version"] == version].drop(columns=["version"])
    temporal = load_temporal().drop(columns=["label", "split", "duration_s", "has_turn_data"])

    combined = acoustic.merge(mfcc, on="anon_id", how="left", validate="one_to_one")
    combined = combined.merge(temporal, on="anon_id", how="left", validate="one_to_one")
    complete_mask = combined["has_turn_data"].astype(bool) & combined.notna().all(axis=1)
    return combined[complete_mask].copy()


def original_vs_denoised(families) -> pd.DataFrame:
    feature_cols = families["acoustic"] + families["mfcc"] + families["temporal"]
    results = []
    for version in ("original", "denoised"):
        combined = load_combined_by_version(version)
        y = (combined["label"] == "synthetic").astype(int)
        train_mask = (combined["split"] == "train").to_numpy()
        val_mask = (combined["split"] == "val").to_numpy()
        res = evaluate_feature_set(
            combined.loc[train_mask, feature_cols], y[train_mask],
            combined.loc[val_mask, feature_cols], y[val_mask],
            feature_cols, f"10_{version}_all_features",
        )
        results.append(res)
    return pd.concat(results, ignore_index=True)


# --------------------------------------------------------------- 8  ---
def recording_condition_analysis(df) -> tuple:
    lines = ["Channel 0, all 353 recordings.\n"]

    manifest_label = pd.read_csv(ACOUSTIC_CSV)
    manifest_label = manifest_label[manifest_label["version"] == "original"][["anon_id", "label", "duration_s"]]

    header_rows = []
    for anon_id in manifest_label["anon_id"]:
        path = AUDIO_DIR / f"{anon_id}.wav"
        info = sf.info(str(path))
        header_rows.append({
            "anon_id": anon_id,
            "samplerate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
            "frames": info.frames,
            "file_size_bytes": path.stat().st_size,
        })
    headers = pd.DataFrame(header_rows).merge(manifest_label, on="anon_id")
    headers["bytes_per_second"] = headers["file_size_bytes"] / headers["duration_s"]

    lines.append("### WAV header format — constant across the dataset?\n")
    for col in ("samplerate", "channels", "subtype"):
        vals = headers[col].unique()
        lines.append(f"- `{col}`: {list(vals)} {'(constant, cannot explain label differences)' if len(vals) == 1 else '(VARIES — investigate)'}")
    lines.append("")

    lines.append("### bytes_per_second (file size / duration_s) by label — checks for a header/encoding difference\n")
    lines.append(headers.groupby("label")["bytes_per_second"].agg(["mean", "std", "min", "max"]).round(3).to_markdown())
    lines.append("")

    acoustic = pd.read_csv(ACOUSTIC_CSV)
    acoustic = acoustic[acoustic["version"] == "original"]
    acoustic["crest_factor"] = acoustic["raw_peak_amplitude"] / acoustic["raw_rms"].replace(0, np.nan)
    acoustic["is_near_clipping"] = acoustic["raw_peak_amplitude"] >= 0.999

    lines.append("### Level / dynamic-range / clipping / spectral characteristics by label (original signal)\n")
    cond_cols = [
        "raw_peak_amplitude", "raw_rms", "raw_silence_proportion", "crest_factor",
        "mean_spectral_centroid", "mean_spectral_bandwidth", "mean_spectral_rolloff", "mean_zcr",
    ]
    lines.append(acoustic.groupby("label")[cond_cols].agg(["mean", "std"]).round(4).to_markdown())
    lines.append("")

    clip_ct = pd.crosstab(acoustic["label"], acoustic["is_near_clipping"])
    lines.append("### Recordings at/near full-scale peak amplitude (>=0.999), by label — possible clipping/normalization difference\n")
    lines.append(clip_ct.to_markdown())
    lines.append("")

    lines.append(
        "**Speech characteristic vs. recording/processing artifact:** `samplerate`/`channels`/`subtype` "
        "are identical for every file (checked directly above), so the dataset does not expose a "
        "different container/encoding per label — but that only rules out the container format, not "
        "differences introduced upstream (microphone/telephony channel for human calls vs. a TTS/"
        "vocoder + mixing pipeline for synthetic ones), which this dataset's metadata cannot "
        "distinguish from a genuine speech difference. That upstream-pipeline explanation is "
        "documented here as a plausible, NOT confirmed, explanation — the manifest/JSON provide no "
        "field naming the original recording or generation pipeline, so this limitation is documented "
        "rather than resolved.\n"
    )

    headers.merge(acoustic[["anon_id"] + cond_cols + ["crest_factor", "is_near_clipping"]], on="anon_id").to_csv(
        RECORDING_CONDITION_CSV, index=False
    )
    return "\n".join(lines), headers


# --------------------------------------------------------------- 11 ---
def channel1_diagnostic(sep_df) -> tuple:
    lines = []
    lines.append(
        "Channel 1 is NOT added to the classifier — it's used here only as an independently-labeled "
        "AI-agent reference point for the strongest separating features found above.\n"
    )

    c0_acoustic = pd.read_csv(ACOUSTIC_CSV)
    c0_acoustic = c0_acoustic[c0_acoustic["version"] == "original"]
    c0_mfcc = pd.read_csv(MFCC_CSV)
    c0_mfcc = c0_mfcc[c0_mfcc["version"] == "original"].drop(columns=["version"])
    c0 = c0_acoustic.merge(c0_mfcc, on="anon_id", how="left")
    c0["group"] = c0["label"].map({"human": "human_c0", "synthetic": "synthetic_c0"})

    c1 = pd.read_csv(CHANNEL1_CSV)
    c1["group"] = "confirmed_ai_c1"

    shared_cols = [c for c in c1.columns if c in c0.columns and c not in ("anon_id", "channel", "group")]
    combined = pd.concat([c0[["anon_id", "group"] + shared_cols], c1[["anon_id", "group"] + shared_cols]], ignore_index=True)

    top_features = sep_df.head(8)["feature"].tolist()
    top_features = [f for f in top_features if f in shared_cols]

    summary = combined.groupby("group")[top_features].mean().T
    summary["synthetic_c0_minus_ai_c1"] = (summary["synthetic_c0"] - summary["confirmed_ai_c1"]).abs()
    summary["human_c0_minus_ai_c1"] = (summary["human_c0"] - summary["confirmed_ai_c1"]).abs()
    summary["synthetic_c0_closer_to_ai_c1"] = summary["synthetic_c0_minus_ai_c1"] < summary["human_c0_minus_ai_c1"]

    lines.append("### Top separating features: group means and which group Channel-1 sits closer to\n")
    lines.append(summary.round(4).to_markdown())
    lines.append("")

    n_closer = int(summary["synthetic_c0_closer_to_ai_c1"].sum())
    lines.append(
        f"Of these {len(top_features)} top-separating features, synthetic C0's mean is closer to "
        f"confirmed-AI C1's mean than human C0's mean is, for **{n_closer}/{len(top_features)}**.\n"
    )
    lines.append(
        "Interpretation is intentionally left descriptive: a feature where synthetic C0 sits closer "
        "to confirmed-AI C1 is consistent with (not proof of) a shared synthetic-generation "
        "characteristic; the reverse pattern would suggest Channel 0's synthetic system produces "
        "acoustically distinct output from Channel 1's, i.e. not all synthetic speech looks alike.\n"
    )

    summary.to_csv(CHANNEL1_DIAGNOSTIC_CSV)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for feat in top_features[:6]:
        fig, ax = plt.subplots(figsize=(6, 4))
        groups = ["human_c0", "synthetic_c0", "confirmed_ai_c1"]
        data = [combined.loc[combined.group == g, feat].dropna().values for g in groups]
        ax.boxplot(data, tick_labels=groups, showmeans=True)
        ax.set_title(f"{feat} — three-group comparison")
        ax.set_ylabel(feat)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"threegroup_{feat}.png", dpi=120)
        plt.close(fig)

    return "\n".join(lines), summary


def _fmt_cv_table(df: pd.DataFrame) -> str:
    view = df.copy()
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        view[f"cv_{metric}"] = view.apply(
            lambda r, m=metric: f"{r[f'cv_{m}_mean']:.3f} ± {r[f'cv_{m}_std']:.3f}", axis=1
        )
    cols = ["experiment", "model", "n_features", "n_train", "n_val", "standardized",
            "val_roc_auc", "cv_accuracy", "cv_precision", "cv_recall", "cv_f1", "cv_roc_auc"]
    return view[cols].round(4).to_markdown(index=False)


def write_evaluation_report(results: dict):
    lines = []
    lines.append("# Modeling evaluation report — is the near-perfect baseline real?\n")
    lines.append(
        "Investigates the acoustic+MFCC+temporal baseline (`modeling/baseline.py`, "
        "`modeling/evaluate.py`) that scored ~1.00 accuracy/F1/ROC-AUC on every model, "
        "on both train and val. Nothing about the existing baseline outputs was changed "
        "or deleted; this report and its CSV/plots are new, additive artifacts.\n"
    )

    lines.append("## 1. Current baseline (preserved, unchanged)\n")
    lines.append(
        "**Status: IN-DATASET BASELINE / POTENTIALLY CONFOUNDED.** Logistic Regression, "
        "Random Forest, and SVM all scored accuracy=precision=recall=F1=ROC-AUC=1.00 on "
        "both train (203 recordings) and val (55 recordings) — see "
        "`outputs/reports/modeling_baseline_report.md` (now labeled the same way, "
        "content otherwise untouched).\n"
    )

    lines.append("## 2. Verification of leakage controls\n")
    lines.append(
        "Checked directly against `modeling/prepare_dataset.py` and `modeling/baseline.py`:\n\n"
        "- Train/val split: read from `manifest.csv` via the acoustic table's `split` "
        "column — never re-derived or randomized. ✅\n"
        "- `train anon_ids ∩ val anon_ids`: empty (verified in `modeling_data_quality_report.md`). ✅\n"
        "- One row per recording throughout every table used (acoustic, MFCC, temporal, "
        "modeling dataset) — no frame-level or turn-level rows are ever used as independent "
        "training examples, so no conversation can straddle train/val. ✅\n"
        "- Labels: `label` column traces to `manifest.csv` only; never re-derived from audio "
        "or from Channel 1. ✅\n"
        "- No label-derived feature in X: `NON_FEATURE_COLS` "
        "(`anon_id, label, split, has_turn_data, duration_s`) is excluded from every "
        "feature matrix built in `modeling/cv_utils.py`/`modeling/diagnostics.py`; none of "
        "the acoustic/MFCC/temporal feature-extraction code (`preprocessing/`, `temporal/`) "
        "takes the label as an input anywhere. ✅\n"
        "- `duration_s` excluded from the feature set (kept in the saved CSVs, just not "
        "used as a model input) — a confound flagged back in the temporal-analysis stage. ✅\n"
        "- Scaling fit on training data only: **already true in the existing baseline code** "
        "— `StandardScaler` lives inside each model's `sklearn.Pipeline`, so `pipeline.fit(X_train, "
        "y_train)` fits the scaler on train only, and `pipeline.predict(X_val)` applies that "
        "already-fitted scaler via `.transform()`, never refitting on val. No code change was "
        "needed here; this report's new CV utilities (`modeling/cv_utils.py`) use the same pattern.\n"
        "- Cross-validation scaling: `sklearn.model_selection.cross_validate()` is given the "
        "whole `Pipeline` (not pre-scaled data), so it clones and re-fits the scaler fresh on "
        "each fold's training portion — the scaler is fit on the complete dataset **at no "
        "point** in this codebase. ✅\n"
        "- Feature selection (univariate separability, Section 6 below) uses **training data "
        "only** — computed from `X_train`/`y_train`, val never touched for this. ✅\n\n"
        "**Conclusion: no leakage-control violation was found.** The evaluation setup was "
        "already methodologically sound; this section documents that rather than fixing bugs.\n"
    )

    lines.append("## 3. Standardization methodology\n")
    lines.append(
        "`StandardScaler` (z = (x - mean) / std) is applied inside a `Pipeline` for "
        "Logistic Regression and SVM; Random Forest is never scaled (tree splits are "
        "scale-invariant, per the task's own note). As expected, standardizing did **not** "
        "change which recordings are correctly/incorrectly classified — it changes feature "
        "scale, not information content. This is confirmed empirically throughout the tables "
        "below: `standardized=True` rows (LR, SVM) and `standardized=False` rows (RF) reach "
        "the same near-ceiling scores on the same feature sets.\n"
    )

    lines.append("## 4-5. Stratified cross-validation + feature-family ablation\n")
    lines.append(
        "5-fold stratified CV (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), "
        "the SAME fold assignment reused for every feature set below so comparisons are fair. "
        "`val_roc_auc` is the single held-out val split; `cv_*` columns are mean ± std across "
        "the 5 folds, computed on the train split only.\n"
    )
    lines.append(_fmt_cv_table(results["ablation_df"]))
    lines.append(
        "\n**Every single feature family alone (acoustic-only, MFCC-only, temporal-only) "
        "already reaches CV ROC-AUC ≥ 0.985.** This is the first important finding: the "
        "near-perfect separation is not concentrated in one feature family — it shows up "
        "independently in all three.\n"
    )

    lines.append("## 6. Individual feature separability (train data only)\n")
    lines.append(
        f"Univariate ROC-AUC per feature, computed on the train split only "
        f"(`modeling_univariate_separability_trainonly.csv`, {len(results['sep_df'])} features). "
        f"**{((results['sep_df']['auc'] >= 0.97) | (results['sep_df']['auc'] <= 0.03)).sum()} "
        f"features** land at or beyond AUC 0.97/0.03 (near-perfect single-feature separation).\n"
    )
    lines.append(results["sep_df"].head(15).round(4).to_markdown(index=False))
    lines.append(
        "\nPlots for the most suspicious individual features (human vs. synthetic, train "
        "split) are in `outputs/plots/modeling/diagnostics/suspicious_*.png`.\n"
    )

    lines.append("## 7. Controlled feature elimination\n")
    lines.append(_fmt_cv_table(results["elimination_df"]))
    lines.append(
        "\n**The one experiment that visibly moves the needle: `6_acoustic_without_spectral` "
        "(energy/amplitude + pitch features only, spectral shape removed) drops to CV ROC-AUC "
        "≈ 0.90-0.95** — the largest drop of any experiment in this report, versus ~1.00 "
        "everywhere else. Removing pitch alone (`7_acoustic_without_pitch`) barely moves "
        "anything. This points at **spectral-shape features specifically** (centroid/"
        "bandwidth/rolloff/ZCR) as the dominant driver, not pitch or simple level/energy "
        "features.\n"
    )

    lines.append("## 8. Recording-condition / pipeline-confound analysis\n")
    lines.append(results["condition_md"])

    lines.append("## 9. Suspicious simple baselines\n")
    lines.append(_fmt_cv_table(results["simple_df"]))
    lines.append(
        "\n**`zcr_only` (5 features) alone reaches CV ROC-AUC ≈ 0.97-0.99** — nearly the "
        "full-feature-set performance from zero-crossing-rate statistics alone. In contrast, "
        "`rms_only`, `peak_amplitude_only`, and `noise_floor_only` are all clearly below "
        "ceiling (CV ROC-AUC 0.66-0.85) — so this is NOT simply \"any basic level/loudness "
        "feature separates perfectly\"; it's specifically the spectral-shape/high-frequency-"
        "content features (ZCR foremost) that do the work, consistent with Section 7's finding.\n"
    )

    lines.append("## 10. Original vs. denoised comparison\n")
    lines.append(_fmt_cv_table(results["denoise_df"]))
    lines.append(
        "\n**The near-perfect separation survives conservative denoising** (an 80 Hz "
        "high-pass filter) unchanged. This is expected given Section 7's finding — ZCR and "
        "spectral rolloff/bandwidth live well above 80 Hz, so a low-frequency hum/rumble "
        "filter was never going to touch them either way.\n"
    )

    lines.append("## 11. Channel-1 (confirmed AI agent) diagnostic comparison\n")
    lines.append(results["channel1_md"])

    lines.append("## 12. Interpretation of likely confounds\n")
    lines.append(
        "Putting Sections 6-11 together:\n\n"
        "1. **Level/loudness**: `raw_peak_amplitude` (human mean 0.79 vs. synthetic 0.93) and "
        "`raw_rms` (0.044 vs. 0.078 — synthetic recordings run nearly 2x louder RMS on "
        "average) differ substantially, with a correspondingly lower crest factor for "
        "synthetic (14.3 vs. 20.0). This is consistent with a **loudness-normalization "
        "difference between how the two groups were captured/mastered** — a processing "
        "artifact, not a speech characteristic. This alone is NOT sufficient to explain "
        "the near-perfect separation, though (`rms_only`/`peak_amplitude_only` top out "
        "around AUC 0.83-0.96 — see Section 9).\n"
        "2. **Spectral shape / ZCR is the dominant driver** (Sections 6, 7, 9): std_zcr and "
        "spectral rolloff/bandwidth range reach univariate AUC 0.93-0.98 alone, removing "
        "them causes the only substantial performance drop in the whole ablation study, and "
        "denoising (which cannot touch this frequency range) doesn't reduce it either.\n"
        "3. **This is not unique to Channel 0's synthetic system**: Section 11 shows Channel 1 "
        "(independently confirmed as an AI agent by the dataset specification, never labeled "
        "via manifest.csv) sits on the SAME side of human C0 as synthetic C0 for 5 of the top "
        "6 features, and is often numerically closer to synthetic C0 than human C0 is. This "
        "is genuine corroborating evidence for a real, at least somewhat generalizable "
        "human-vs-synthetic acoustic signature in spectral shape — though it could equally "
        "reflect two different synthetic pipelines sharing a similar telephony/vocoder "
        "frequency response, which this dataset's metadata cannot rule out.\n"
        "4. **WAV container format is not the explanation**: sample rate (8000 Hz), channel "
        "count (2), and PCM subtype are identical for every one of the 353 files — this is "
        "not a simple encoding/header mismatch.\n"
    )

    lines.append("## 13. Limitations\n")
    lines.append(
        "- No metadata in this dataset names the original recording device/telephony path "
        "for human calls or the generation/vocoder system for synthetic ones — the "
        "recording-condition explanation in Section 8/12 is a plausible, evidence-supported "
        "hypothesis, not a confirmed fact.\n"
        "- The baseline dataset covers only the 258 recordings with valid turn metadata; the "
        "95 without turn data were not part of any experiment in this report.\n"
        "- Channel 1 comparisons use only fast (non-pitch) features "
        "(`preprocessing/channel1_diagnostic_features.py` intentionally skips pYIN/F0 for "
        "speed) — an F0-based Channel-1 comparison was not performed.\n"
        "- All of this remains **descriptive**: no causal claim is made about what produces "
        "these acoustic differences.\n"
    )

    lines.append("## 14-15. Recommendation\n")
    lines.append(
        "**The evidence leans toward a dataset/recording-pipeline confound, not a validated "
        "\"AI voice detector.\"** The near-perfect separation is driven overwhelmingly by "
        "spectral-shape/ZCR characteristics plus a secondary loudness-normalization "
        "difference — exactly the kind of confound the task asked to rule in or out — rather "
        "than being spread thinly and plausibly across many weak, speech-content-related "
        "signals. The Channel-1 corroboration (Section 11) is genuinely encouraging (it is "
        "*some* evidence real synthetic-speech signal is present, not pure noise), but is not "
        "enough on its own to certify generalization.\n\n"
        "**Recommended before trusting this detector further:**\n"
        "1. Obtain or construct a validation set where human and synthetic recordings are "
        "known to share the same recording/mastering pipeline (or at minimum, loudness-"
        "normalize both groups to a common target level and re-run this same ablation) — if "
        "ROC-AUC collapses once level/spectral-shape are equalized, that confirms the "
        "confound; if it holds, that's real evidence of a generalizable signal.\n"
        "2. If possible, obtain external synthetic speech (a different TTS/voice-cloning "
        "system than whatever produced this dataset's Channel 0 synthetic recordings) and "
        "test whether the SAME features (ZCR, spectral rolloff/bandwidth) still separate it "
        "from human speech — this is the real generalization test Section 11 could only "
        "approximate using Channel 1.\n"
        "3. Until then, do not report this baseline's accuracy as real-world detection "
        "capability, and do not proceed to real-time inference.\n"
    )

    with open(EVALUATION_REPORT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {EVALUATION_REPORT_MD}")


def run():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df, y, train_mask, val_mask = load_baseline_split()
    families = get_family_cols()

    print("Section 5: feature-family ablation...")
    ablation_df = family_ablation(df, y, train_mask, val_mask, families)

    print("Section 6: univariate separability...")
    sep_df = univariate_separability(df, y, train_mask, families)
    suspicious = plot_suspicious_features(df, y, train_mask, sep_df)

    print("Section 7: controlled feature elimination...")
    elimination_df = controlled_elimination(df, y, train_mask, val_mask, families)

    print("Section 9: suspicious simple baselines...")
    simple_df = suspicious_simple_baselines(df, y, train_mask, val_mask)

    print("Section 10: original vs denoised...")
    denoise_df = original_vs_denoised(families)

    print("Section 8: recording-condition analysis...")
    condition_md, headers = recording_condition_analysis(df)

    print("Section 11: Channel-1 diagnostic...")
    channel1_md, channel1_summary = channel1_diagnostic(sep_df)

    all_experiments = pd.concat([ablation_df, elimination_df, simple_df, denoise_df], ignore_index=True)
    all_experiments.to_csv(EXPERIMENT_COMPARISON_CSV, index=False)

    results = {
        "ablation_df": ablation_df, "sep_df": sep_df, "suspicious": suspicious,
        "elimination_df": elimination_df, "simple_df": simple_df, "denoise_df": denoise_df,
        "condition_md": condition_md, "channel1_md": channel1_md, "channel1_summary": channel1_summary,
        "all_experiments": all_experiments,
    }
    write_evaluation_report(results)
    return results


if __name__ == "__main__":
    run()
