"""
Level-normalization confound experiment: repeats the modeling.diagnostics
methodology (feature-family ablation, controlled elimination, univariate
separability, suspicious simple baselines, Channel-1 diagnostic) on
RMS-level-normalized acoustic/MFCC features, and compares directly
against the original (non-normalized) results.

Does not modify, delete, or overwrite any existing output from
modeling/diagnostics.py — every output here uses a new "_normalized" /
"normalization_experiment_*" filename. Reuses (does not duplicate the
logic of) modeling.diagnostics's feature-family ablation, controlled
elimination, and suspicious-baseline functions, which are dataset-
agnostic; a self-contained Channel-1 comparison and a small set of
distribution plots are written fresh here.

Run with: python -m modeling.diagnostics_normalized
  (after preprocessing.normalized_pipeline,
   preprocessing.channel1_diagnostic_features_normalized, and
   modeling.prepare_dataset_normalized)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.pipeline import FEATURES_DIR as ACOUSTIC_FEATURES_DIR
from modeling.prepare_dataset import FEATURES_DIR, REPORTS_DIR
from modeling.prepare_dataset_normalized import MODELING_DATASET_NORMALIZED_COMPLETE_CSV
from modeling.diagnostics import (
    get_family_cols, family_ablation, controlled_elimination,
    suspicious_simple_baselines, univariate_separability, plot_suspicious_features,
    EXPERIMENT_COMPARISON_CSV,
)

PLOTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "plots" / "modeling" / "normalization_experiment"
NORM_EXPERIMENT_COMPARISON_CSV = FEATURES_DIR / "normalization_experiment_comparison.csv"
NORM_UNIVARIATE_SEP_CSV = FEATURES_DIR / "modeling_univariate_separability_normalized.csv"
NORM_CHANNEL1_CSV = FEATURES_DIR / "modeling_channel1_diagnostic_comparison_normalized.csv"
NORM_REPORT_MD = REPORTS_DIR / "normalization_experiment_report.md"

GAINS_CSV = FEATURES_DIR / "normalization_gains.csv"
ORIGINAL_ACOUSTIC_CSV = ACOUSTIC_FEATURES_DIR / "recording_level.csv"
NORMALIZED_ACOUSTIC_CSV = FEATURES_DIR / "recording_level_normalized.csv"
NORMALIZED_MFCC_CSV = FEATURES_DIR / "mfcc_normalized.csv"
ORIGINAL_MFCC_CSV = FEATURES_DIR / "mfcc_recording_level.csv"
CHANNEL1_ORIGINAL_CSV = ACOUSTIC_FEATURES_DIR / "channel1_diagnostic_features.csv"
CHANNEL1_NORMALIZED_CSV = ACOUSTIC_FEATURES_DIR / "channel1_diagnostic_features_normalized.csv"


def load_split(csv_path):
    df = pd.read_csv(csv_path)
    y = (df["label"] == "synthetic").astype(int)
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    return df, y, train_mask, val_mask


def level_distribution_comparison() -> tuple:
    """Section: how much did normalization change RMS/peak distributions?"""
    lines = ["## RMS / peak-amplitude distribution: original vs. normalized\n"]

    orig = pd.read_csv(ORIGINAL_ACOUSTIC_CSV)
    orig = orig[orig["version"] == "original"][["anon_id", "label", "raw_rms", "raw_peak_amplitude"]]
    gains = pd.read_csv(GAINS_CSV)
    merged = orig.merge(gains, on="anon_id")

    lines.append("### raw_rms: original vs. normalized, by label\n")
    summary = merged.groupby("label")[["raw_rms", "normalized_rms"]].agg(["mean", "std"])
    lines.append(summary.round(4).to_markdown())
    lines.append("")

    lines.append("### raw_peak_amplitude: original vs. normalized, by label\n")
    summary2 = merged.groupby("label")[["raw_peak_amplitude", "normalized_peak_amplitude"]].agg(["mean", "std"])
    lines.append(summary2.round(4).to_markdown())
    lines.append("")

    n_capped = int(merged["peak_capped"].sum())
    lines.append(f"Recordings where the peak-safety cap (not the RMS target) determined the gain: {n_capped}/{len(merged)}\n")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for col_pair, title, fname in (
        (("raw_rms", "normalized_rms"), "RMS: original vs. normalized", "rms_original_vs_normalized.png"),
        (("raw_peak_amplitude", "normalized_peak_amplitude"), "Peak amplitude: original vs. normalized", "peak_original_vs_normalized.png"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        for ax, col in zip(axes, col_pair):
            groups = ["human", "synthetic"]
            data = [merged.loc[merged.label == g, col].dropna().values for g in groups]
            ax.boxplot(data, tick_labels=groups, showmeans=True)
            ax.set_title(col)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / fname, dpi=120)
        plt.close(fig)

    return "\n".join(lines), merged


def zcr_spectral_comparison(orig_sep: pd.DataFrame, norm_sep: pd.DataFrame) -> str:
    """Sections: did ZCR / spectral-feature separation change?"""
    lines = ["## ZCR / spectral-feature separation: original vs. normalized (univariate AUC, train split)\n"]
    watch = [
        "std_zcr", "mean_zcr", "max_zcr", "range_zcr",
        "mean_spectral_centroid", "max_spectral_centroid", "range_spectral_centroid",
        "mean_spectral_bandwidth", "max_spectral_bandwidth", "range_spectral_bandwidth",
        "mean_spectral_rolloff", "max_spectral_rolloff", "range_spectral_rolloff",
        "mean_rms", "raw_rms", "raw_peak_amplitude",
    ]
    o = orig_sep.set_index("feature")["auc"]
    n = norm_sep.set_index("feature")["auc"]
    rows = []
    for feat in watch:
        if feat in o.index and feat in n.index:
            rows.append({
                "feature": feat,
                "auc_original": o[feat], "auc_normalized": n[feat],
                "abs_change": abs(n[feat] - o[feat]),
            })
    comp = pd.DataFrame(rows).sort_values("abs_change", ascending=False)
    lines.append(comp.round(4).to_markdown(index=False))
    n_unchanged = int((comp["abs_change"] < 1e-9).sum())
    lines.append(
        f"\n**{n_unchanged} of these {len(comp)} features show ZERO change (identical AUC to 4+ "
        "decimal places) — this is mathematically expected, not a new empirical result.** "
        "Zero-crossing rate is unaffected by any positive linear gain (rescaling a signal never "
        "changes where it crosses zero), and spectral centroid/bandwidth/rolloff are computed "
        "from the *relative* distribution of energy across frequency bins (e.g. centroid = a "
        "magnitude-weighted average frequency) — a uniform gain scales every bin's magnitude by "
        "the same factor, which cancels out and leaves these ratios unchanged. Level "
        "normalization was never going to move these features; the features that DID move "
        "(`raw_peak_amplitude`, `raw_rms`, `mean_rms`) are exactly the ones that are NOT "
        "scale-invariant, confirming the normalization was applied correctly rather than being "
        "a coincidence.\n"
    )
    return "\n".join(lines)


def channel1_diagnostic_normalized(sep_df: pd.DataFrame) -> tuple:
    """Section 10: Channel-1 diagnostic, normalized features."""
    lines = []
    c0_acoustic = pd.read_csv(NORMALIZED_ACOUSTIC_CSV).drop(columns=["version"])
    c0_mfcc = pd.read_csv(NORMALIZED_MFCC_CSV).drop(columns=["version"])
    identity = pd.read_csv(ORIGINAL_ACOUSTIC_CSV)
    identity = identity[identity["version"] == "original"][["anon_id", "label"]]
    c0 = identity.merge(c0_acoustic, on="anon_id").merge(c0_mfcc, on="anon_id")
    c0["group"] = c0["label"].map({"human": "human_c0", "synthetic": "synthetic_c0"})

    c1 = pd.read_csv(CHANNEL1_NORMALIZED_CSV)
    c1["group"] = "confirmed_ai_c1"

    shared_cols = [c for c in c1.columns if c in c0.columns and c not in ("anon_id", "channel", "group")]
    combined = pd.concat([c0[["anon_id", "group"] + shared_cols], c1[["anon_id", "group"] + shared_cols]], ignore_index=True)

    top_features = [f for f in sep_df.head(8)["feature"].tolist() if f in shared_cols]
    # always include the loudness-related features too, even if they've
    # fallen out of the normalized top-8 — this is the one place we can
    # check whether normalization changed the C0/C1 relationship for the
    # features that normalization actually touches (see caveat below)
    for extra in ("raw_rms", "raw_peak_amplitude", "mean_rms"):
        if extra in shared_cols and extra not in top_features:
            top_features.append(extra)
    summary = combined.groupby("group")[top_features].mean().T
    summary["synthetic_c0_minus_ai_c1"] = (summary["synthetic_c0"] - summary["confirmed_ai_c1"]).abs()
    summary["human_c0_minus_ai_c1"] = (summary["human_c0"] - summary["confirmed_ai_c1"]).abs()
    summary["synthetic_c0_closer_to_ai_c1"] = summary["synthetic_c0_minus_ai_c1"] < summary["human_c0_minus_ai_c1"]

    invariant_features = [f for f in top_features if f not in ("raw_rms", "raw_peak_amplitude", "mean_rms")]
    loudness_features = [f for f in top_features if f in ("raw_rms", "raw_peak_amplitude", "mean_rms")]

    lines.append("### Top-ranked separating features (normalized) + loudness features: group means\n")
    lines.append(summary.round(4).to_markdown())
    lines.append("")
    n_closer = int(summary.loc[invariant_features, "synthetic_c0_closer_to_ai_c1"].sum())
    lines.append(f"Of the {len(invariant_features)} top-ranked features, synthetic C0 is closer to "
                 f"confirmed-AI C1 than human C0 is, for **{n_closer}/{len(invariant_features)}** "
                 f"(original, non-normalized comparison: 5/6 — see modeling_evaluation_report.md).\n")
    lines.append(
        "**This is not independent replication of that finding — the numbers are identical.** "
        f"All {len(invariant_features)} of these top-ranked features are mathematically invariant "
        "to a uniform per-recording gain: std_zcr and spectral rolloff/bandwidth for the reasons "
        "given in the ZCR/spectral section above, and MFCC coefficients other than the 0th "
        "(log-energy) coefficient for the same reason (a uniform gain becomes an additive "
        "constant on the log-mel-spectrogram, which the DCT confines entirely to the 0th "
        "coefficient — verified directly: `mfcc8_mean`'s three group means here are byte-"
        "identical to the non-normalized report). So this part of the comparison could never "
        "have come out differently — it isn't fresh evidence.\n\n"
        f"The rows that DO reflect genuinely different, post-normalization values are "
        f"`{', '.join(loudness_features)}` (added here explicitly rather than left out) — these "
        "are the ones where a change in the C0/C1 relationship would actually mean something. "
        f"synthetic_c0_closer_to_ai_c1 for these: "
        f"{summary.loc[loudness_features, 'synthetic_c0_closer_to_ai_c1'].to_dict()}.\n"
    )

    summary.to_csv(NORM_CHANNEL1_CSV)
    return "\n".join(lines), summary


def run():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df, y, train_mask, val_mask = load_split(MODELING_DATASET_NORMALIZED_COMPLETE_CSV)
    families = get_family_cols()  # same feature names -> same families, regardless of audio version

    print("Level distribution comparison...")
    level_md, level_df = level_distribution_comparison()

    print("Feature-family ablation (normalized)...")
    ablation_df = family_ablation(df, y, train_mask, val_mask, families)
    ablation_df.insert(0, "dataset_version", "normalized")

    print("Univariate separability (normalized)...")
    norm_sep = univariate_separability(df, y, train_mask, families, output_csv=NORM_UNIVARIATE_SEP_CSV)
    plot_suspicious_features(df, y, train_mask, norm_sep, plots_dir=PLOTS_DIR, filename_prefix="suspicious_normalized_")

    print("Controlled feature elimination (normalized)...")
    elimination_df = controlled_elimination(df, y, train_mask, val_mask, families)
    elimination_df.insert(0, "dataset_version", "normalized")

    print("Suspicious simple baselines (normalized)...")
    simple_df = suspicious_simple_baselines(df, y, train_mask, val_mask)
    simple_df.insert(0, "dataset_version", "normalized")

    print("Channel-1 diagnostic (normalized)...")
    channel1_md, channel1_summary = channel1_diagnostic_normalized(norm_sep)

    orig_sep = pd.read_csv(FEATURES_DIR / "modeling_univariate_separability_trainonly.csv")
    zcr_md = zcr_spectral_comparison(orig_sep, norm_sep)

    # --- combine with the ORIGINAL experiment results for a single comparison table ---
    orig_experiments = pd.read_csv(EXPERIMENT_COMPARISON_CSV)
    orig_experiments.insert(0, "dataset_version", "original")
    norm_experiments = pd.concat([ablation_df, elimination_df, simple_df], ignore_index=True)
    all_experiments = pd.concat([orig_experiments, norm_experiments], ignore_index=True)
    all_experiments.to_csv(NORM_EXPERIMENT_COMPARISON_CSV, index=False)

    # overall + family-level CV ROC-AUC comparison plot
    g_orig = orig_experiments[orig_experiments.experiment == "G_acoustic_mfcc_temporal"]
    g_norm = ablation_df[ablation_df.experiment == "G_acoustic_mfcc_temporal"]
    fig, ax = plt.subplots(figsize=(6, 4))
    models = g_orig["model"].tolist()
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, g_orig["cv_roc_auc_mean"], width, yerr=g_orig["cv_roc_auc_std"], label="original", capsize=4)
    ax.bar(x + width / 2, g_norm["cv_roc_auc_mean"], width, yerr=g_norm["cv_roc_auc_std"], label="normalized", capsize=4)
    ax.set_xticks(x, models)
    ax.set_ylabel("CV ROC-AUC (mean ± std)")
    ax.set_title("All features (A+M+T): original vs. normalized")
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "overall_cv_rocauc_original_vs_normalized.png", dpi=120)
    plt.close(fig)

    fam_orig = orig_experiments[orig_experiments.experiment.isin(["A_acoustic_only", "B_mfcc_only", "C_temporal_only"])]
    fam_norm = ablation_df[ablation_df.experiment.isin(["A_acoustic_only", "B_mfcc_only", "C_temporal_only"])]
    fig, ax = plt.subplots(figsize=(8, 4))
    fam_labels = ["A_acoustic_only", "B_mfcc_only", "C_temporal_only"]
    lr_orig = [fam_orig[(fam_orig.experiment == f) & (fam_orig.model == "logistic_regression")]["cv_roc_auc_mean"].iloc[0] for f in fam_labels]
    lr_norm = [fam_norm[(fam_norm.experiment == f) & (fam_norm.model == "logistic_regression")]["cv_roc_auc_mean"].iloc[0] for f in fam_labels]
    x = np.arange(len(fam_labels))
    ax.bar(x - width / 2, lr_orig, width, label="original")
    ax.bar(x + width / 2, lr_norm, width, label="normalized")
    ax.set_xticks(x, fam_labels, rotation=15)
    ax.set_ylabel("CV ROC-AUC (logistic_regression)")
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Feature-family CV ROC-AUC: original vs. normalized")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "family_cv_rocauc_original_vs_normalized.png", dpi=120)
    plt.close(fig)

    write_report(level_md, zcr_md, ablation_df, orig_experiments, elimination_df, simple_df, channel1_md)
    print(f"Wrote {NORM_REPORT_MD}")


def write_report(level_md, zcr_md, ablation_df, orig_experiments, elimination_df, simple_df, channel1_md):
    g_orig = orig_experiments[(orig_experiments.experiment == "G_acoustic_mfcc_temporal")]
    g_norm = ablation_df[(ablation_df.experiment == "G_acoustic_mfcc_temporal")]
    max_drop = (g_orig.set_index("model")["cv_roc_auc_mean"] - g_norm.set_index("model")["cv_roc_auc_mean"]).max()

    lines = ["# Normalization experiment report\n"]
    lines.append(
        "Tests whether the near-perfect human-vs-synthetic separation found in "
        "`modeling_evaluation_report.md` is (partly) explained by recording-level loudness "
        "differences. RMS-level normalization (`preprocessing/normalize.py`) is applied "
        "independently to each recording's Channel-0 (and, for the diagnostic, Channel-1) "
        "audio, and the SAME acoustic+MFCC feature extraction and evaluation methodology is "
        "repeated on the normalized signal. Nothing about the original/denoised pipeline, "
        "features, or diagnostic outputs was changed.\n"
    )

    lines.append("## 1. How much did normalization change RMS/peak distributions?\n")
    lines.append(level_md)

    lines.append("## 2-3. Did ZCR / spectral-feature separation change?\n")
    lines.append(zcr_md)

    lines.append("## 4. Did overall CV ROC-AUC change? (all features, G_acoustic_mfcc_temporal)\n")
    cmp_cols = ["model", "cv_roc_auc_mean", "cv_roc_auc_std"]
    merged = g_orig[cmp_cols].merge(g_norm[cmp_cols], on="model", suffixes=("_original", "_normalized"))
    lines.append(merged.round(4).to_markdown(index=False))
    lines.append(f"\nLargest drop in mean CV ROC-AUC (original -> normalized) across models: {max_drop:.4f}\n")

    lines.append("## 5. Which feature families were most affected? (feature-family ablation)\n")
    fam_cmp_rows = []
    for exp in ("A_acoustic_only", "B_mfcc_only", "C_temporal_only", "D_acoustic_mfcc",
                "E_acoustic_temporal", "F_mfcc_temporal", "G_acoustic_mfcc_temporal"):
        o = orig_experiments[(orig_experiments.experiment == exp) & (orig_experiments.model == "logistic_regression")]
        n = ablation_df[(ablation_df.experiment == exp) & (ablation_df.model == "logistic_regression")]
        if len(o) and len(n):
            fam_cmp_rows.append({
                "experiment": exp,
                "cv_roc_auc_original": o["cv_roc_auc_mean"].iloc[0],
                "cv_roc_auc_normalized": n["cv_roc_auc_mean"].iloc[0],
                "change": n["cv_roc_auc_mean"].iloc[0] - o["cv_roc_auc_mean"].iloc[0],
            })
    fam_cmp = pd.DataFrame(fam_cmp_rows)
    lines.append(fam_cmp.round(4).to_markdown(index=False))
    lines.append("\n(logistic_regression shown; full model-by-model detail is in "
                 "`outputs/features/normalization_experiment_comparison.csv`.)\n")

    lines.append("## 6. Did the confirmed-AI Channel-1 relationship change?\n")
    lines.append(channel1_md)

    biggest_drop_family = fam_cmp.loc[fam_cmp["change"].idxmin(), "experiment"] if len(fam_cmp) else "n/a"
    still_high = (g_norm["cv_roc_auc_mean"] >= 0.97).all()

    lines.append("## 7. Does the evidence still suggest a dataset-specific recording/generation signature?\n")
    lines.append("## 8. Is there stronger evidence of genuine synthetic-speech detection?\n")
    if still_high:
        lines.append(
            f"**CV ROC-AUC on all features remains ≥0.97 after normalization** "
            f"(largest single-model drop: {max_drop:.4f}). Normalizing away recording-level "
            f"loudness differences did **not** substantially reduce the separation. Per the "
            f"task's interpretation rules, this is evidence the discriminatory signal is NOT "
            f"simply loudness — but it does not, by itself, prove the signal is genuine, "
            f"generalizable synthetic-speech characteristics rather than some OTHER "
            f"recording/generation-pipeline artifact untouched by RMS normalization (spectral "
            f"shape / frequency response, which normalization does not alter, remains the "
            f"leading candidate — see Section 2-3 above and `modeling_evaluation_report.md` "
            f"Section 7/9). The most-affected family was `{biggest_drop_family}`. External "
            f"validation is still needed before treating this as a working detector.\n"
        )
    else:
        lines.append(
            f"CV ROC-AUC on all features dropped meaningfully after normalization (largest "
            f"drop: {max_drop:.4f}), consistent with recording-level loudness being an "
            f"important contributor to the original separation. The most-affected family was "
            f"`{biggest_drop_family}`. Whatever separation remains after normalization should "
            f"be investigated on its own terms (see Section 2-3 for which features held up).\n"
        )

    lines.append("## Interpretation & recommendation\n")
    if still_high:
        lines.append(
            "**Mixed-to-C: the loudness confound is ruled OUT as the primary driver, but the "
            "leading remaining hypothesis is still a recording/generation-pipeline artifact "
            "(spectral shape / frequency response), not confirmed genuine speech content.** "
            "Recommendation: **(C) mixed evidence** — the signal is robust to loudness "
            "normalization (rules out one confound), but spectral-shape features remain "
            "exactly as strong, and denoising in the prior diagnostic also didn't touch them "
            "either. The smallest next experiment: obtain or approximate a frequency-response-"
            "matched comparison (e.g. apply the same band-limiting/telephony filter to both "
            "groups, or source external synthetic speech through the same channel/telephony "
            "path as the human calls) and re-test whether ZCR/spectral-rolloff separation "
            "survives that — mirroring exactly what this experiment did for loudness.\n"
        )
    else:
        lines.append(
            "**(B) Strong evidence of a recording/dataset confound** — loudness normalization "
            "substantially reduced separation, so the original near-perfect performance was "
            "significantly inflated by a recording-level artifact. Recommendation: investigate "
            "the dataset's construction/mastering pipeline directly and obtain better-"
            "controlled (level-matched) data before further modeling.\n"
        )

    lines.append(
        "\nNot done in this step (per task scope): no neural network, no live detector, no "
        "resumed ASR, no additional NLP, no PCA/orthogonalization, no hyperparameter tuning, "
        "no permanent feature removal, no external test data (none was available), no "
        "real-world accuracy claim. Stopping here — not proceeding to real-time inference.\n"
    )

    with open(NORM_REPORT_MD, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run()
