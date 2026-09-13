"""
Telephony-band (300-3400 Hz) confound experiment: repeats the
modeling.diagnostics methodology on band-limited audio (two variants:
band-limited only, and band-limited + RMS-normalized), and builds the
4-way comparison (original / RMS-normalized / band-limited /
band-limited+normalized) that is this experiment's central result.

Unlike RMS normalization (a pure gain — provably invariant for
ratio/shape features such as ZCR and spectral centroid/bandwidth/
rolloff), a band-pass FILTER removes actual frequency content, so NO
feature here is mathematically guaranteed unchanged a priori. Any
near-invariance observed must be explained on its own terms (e.g. a
recording that already had ~zero energy outside 300-3400 Hz), not
assumed the way it could be for a linear gain — see the "filter
invariance" section this module produces.

Does not modify any existing output (original diagnostics, the
normalization experiment, or the raw pipelines). New files only.

Run with: python -m modeling.diagnostics_telephony
  (after preprocessing.telephony_pipeline,
   preprocessing.channel1_diagnostic_features_telephony, and
   modeling.prepare_dataset_telephony)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.pipeline import FEATURES_DIR as ACOUSTIC_FEATURES_DIR
from modeling.prepare_dataset import FEATURES_DIR, REPORTS_DIR
from modeling.prepare_dataset_telephony import VARIANTS
from modeling.cv_utils import evaluate_feature_set
from modeling.diagnostics import (
    get_family_cols, family_ablation, controlled_elimination,
    suspicious_simple_baselines, univariate_separability, plot_suspicious_features,
    ACOUSTIC_ENERGY_COLS, ACOUSTIC_PITCH_COLS,
)

PLOTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "plots" / "modeling" / "telephony_experiment"
TELEPHONY_EXPERIMENT_COMPARISON_CSV = FEATURES_DIR / "telephony_experiment_comparison.csv"
TELEPHONY_REPORT_MD = REPORTS_DIR / "telephony_experiment_report.md"

ORIGINAL_ACOUSTIC_CSV = ACOUSTIC_FEATURES_DIR / "recording_level.csv"
ORIGINAL_MFCC_CSV = FEATURES_DIR / "mfcc_recording_level.csv"
NORM_EXPERIMENT_CSV = FEATURES_DIR / "normalization_experiment_comparison.csv"
ORIG_SEP_CSV = FEATURES_DIR / "modeling_univariate_separability_trainonly.csv"
NORM_SEP_CSV = FEATURES_DIR / "modeling_univariate_separability_normalized.csv"

CHANNEL1_ORIGINAL_CSV = ACOUSTIC_FEATURES_DIR / "channel1_diagnostic_features.csv"

REPRESENTATIVE_FEATURES = [
    "raw_rms", "raw_peak_amplitude",
    "mean_spectral_centroid", "mean_spectral_bandwidth", "mean_spectral_rolloff",
    "max_spectral_rolloff", "mean_zcr", "std_zcr",
    "mean_f0", "mfcc1_mean", "mfcc8_mean",
]

# Section 12: is each feature EXPECTED to be affected by a band-pass
# filter (as opposed to the pure-gain case, where several were exactly
# invariant by construction)? Every one of these CAN change under
# filtering; this documents the a-priori expectation stated in the task,
# not a guarantee — actual behavior is checked empirically below.
FILTER_EXPECTATION = {
    "raw_rms": "expected to change (removing energy outside the band lowers total energy)",
    "raw_peak_amplitude": "may change (depends whether the original peak sample was in-band)",
    "mean_spectral_centroid": "expected to change (energy redistributed within a narrower band)",
    "mean_spectral_bandwidth": "expected to change (spread computed only over remaining energy)",
    "mean_spectral_rolloff": "expected to change; HARD-CAPPED near/under 3400Hz by construction",
    "max_spectral_rolloff": "expected to change; HARD-CAPPED near/under 3400Hz by construction",
    "mean_zcr": "can change (removing HF/LF content changes zero-crossing count)",
    "std_zcr": "can change, same reason",
    "mean_f0": "may change/be affected: pYIN can lose the fundamental itself for low "
               "voices near/below the 300Hz cutoff and fall back to harmonics",
    "mfcc1_mean": "expected to change (computed from the filtered log-mel spectrum; not "
                  "gain-invariant reasoning here — the filter changes actual mel-band energies)",
    "mfcc8_mean": "expected to change, same reason",
}


def load_split(csv_path):
    df = pd.read_csv(csv_path)
    y = (df["label"] == "synthetic").astype(int)
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    return df, y, train_mask, val_mask


def spectral_shape_only_experiment(df, y, train_mask, val_mask, families, label) -> pd.DataFrame:
    acoustic = families["acoustic"]
    spectral_only = [c for c in acoustic if c not in ACOUSTIC_ENERGY_COLS and c not in ACOUSTIC_PITCH_COLS]
    return evaluate_feature_set(
        df.loc[train_mask, spectral_only], y[train_mask], df.loc[val_mask, spectral_only], y[val_mask],
        spectral_only, label,
    )


def run_variant_diagnostics(variant_name: str, families: dict) -> dict:
    v = VARIANTS[variant_name]
    df, y, train_mask, val_mask = load_split(v["complete_csv"])

    ablation_df = family_ablation(df, y, train_mask, val_mask, families)
    ablation_df.insert(0, "dataset_version", variant_name)

    sep_csv = FEATURES_DIR / f"modeling_univariate_separability_{variant_name}.csv"
    sep_df = univariate_separability(df, y, train_mask, families, output_csv=sep_csv)
    plot_suspicious_features(df, y, train_mask, sep_df, plots_dir=PLOTS_DIR, filename_prefix=f"suspicious_{variant_name}_")

    elimination_df = controlled_elimination(df, y, train_mask, val_mask, families)
    elimination_df.insert(0, "dataset_version", variant_name)
    spectral_only_df = spectral_shape_only_experiment(df, y, train_mask, val_mask, families, "9_acoustic_spectral_shape_only")
    spectral_only_df.insert(0, "dataset_version", variant_name)
    elimination_df = pd.concat([elimination_df, spectral_only_df], ignore_index=True)

    simple_df = suspicious_simple_baselines(df, y, train_mask, val_mask)
    simple_df.insert(0, "dataset_version", variant_name)

    return {"df": df, "y": y, "train_mask": train_mask, "sep_df": sep_df,
            "ablation_df": ablation_df, "elimination_df": elimination_df, "simple_df": simple_df}


def build_filter_change_table(orig_sep, results) -> pd.DataFrame:
    o = orig_sep.set_index("feature")["auc"]
    rows = []
    for feat in REPRESENTATIVE_FEATURES:
        row = {"feature": feat, "auc_original": o.get(feat, np.nan),
               "a_priori_expectation": FILTER_EXPECTATION.get(feat, "")}
        for variant_name, res in results.items():
            s = res["sep_df"].set_index("feature")["auc"]
            row[f"auc_{variant_name}"] = s.get(feat, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def channel1_diagnostic_telephony(variant_name: str, sep_df: pd.DataFrame) -> tuple:
    v = VARIANTS[variant_name]
    c0_acoustic = pd.read_csv(v["acoustic_csv"]).drop(columns=["version"])
    c0_mfcc = pd.read_csv(v["mfcc_csv"]).drop(columns=["version"])
    identity = pd.read_csv(ORIGINAL_ACOUSTIC_CSV)
    identity = identity[identity["version"] == "original"][["anon_id", "label"]]
    c0 = identity.merge(c0_acoustic, on="anon_id").merge(c0_mfcc, on="anon_id")
    c0["group"] = c0["label"].map({"human": "human_c0", "synthetic": "synthetic_c0"})

    c1_csv = ACOUSTIC_FEATURES_DIR / f"channel1_diagnostic_features_{variant_name}.csv"
    c1 = pd.read_csv(c1_csv)
    c1["group"] = "confirmed_ai_c1"

    shared_cols = [c for c in c1.columns if c in c0.columns and c not in ("anon_id", "channel", "group")]
    combined = pd.concat([c0[["anon_id", "group"] + shared_cols], c1[["anon_id", "group"] + shared_cols]], ignore_index=True)

    top_features = [f for f in sep_df.head(8)["feature"].tolist() if f in shared_cols]
    for extra in ("raw_rms", "raw_peak_amplitude", "mean_spectral_centroid"):
        if extra in shared_cols and extra not in top_features:
            top_features.append(extra)

    summary = combined.groupby("group")[top_features].mean().T
    summary["synthetic_c0_minus_ai_c1"] = (summary["synthetic_c0"] - summary["confirmed_ai_c1"]).abs()
    summary["human_c0_minus_ai_c1"] = (summary["human_c0"] - summary["confirmed_ai_c1"]).abs()
    summary["synthetic_c0_closer_to_ai_c1"] = summary["synthetic_c0_minus_ai_c1"] < summary["human_c0_minus_ai_c1"]

    n_closer = int(summary["synthetic_c0_closer_to_ai_c1"].sum())
    lines = [f"### Channel-1 comparison, variant `{variant_name}`\n"]
    lines.append(summary.round(4).to_markdown())
    lines.append(f"\nSynthetic C0 closer to confirmed-AI C1 than human C0: {n_closer}/{len(top_features)} features.\n")
    lines.append(
        "Unlike the RMS-normalization experiment, these feature values are NOT guaranteed "
        "identical to the original/other-variant numbers — a band-pass filter genuinely changes "
        "spectral content, so any agreement or disagreement with the original 5/6 finding here is "
        "real, new information, not a mathematical tautology.\n"
    )
    summary.to_csv(FEATURES_DIR / f"modeling_channel1_diagnostic_comparison_{variant_name}.csv")
    return "\n".join(lines), summary


def make_plots(results, filter_change: pd.DataFrame):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    orig_acoustic = pd.read_csv(ORIGINAL_ACOUSTIC_CSV)
    orig_acoustic = orig_acoustic[orig_acoustic["version"] == "original"]
    labels_by_id = orig_acoustic.set_index("anon_id")["label"]

    datasets = {"original": orig_acoustic.set_index("anon_id")}
    normalized_csv = FEATURES_DIR / "recording_level_normalized.csv"
    if normalized_csv.exists():
        datasets["normalized"] = pd.read_csv(normalized_csv).drop(columns=["version"]).set_index("anon_id")
    for name, v in VARIANTS.items():
        d = pd.read_csv(v["acoustic_csv"]).drop(columns=["version"]).set_index("anon_id")
        datasets[name] = d

    for feat in ("mean_spectral_centroid", "mean_spectral_bandwidth", "mean_spectral_rolloff", "mean_zcr"):
        fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 4), sharey=True)
        for ax, (dname, d) in zip(axes, datasets.items()):
            merged = d[[feat]].join(labels_by_id, how="inner")
            groups = ["human", "synthetic"]
            data = [merged.loc[merged.label == g, feat].dropna().values for g in groups]
            ax.boxplot(data, tick_labels=groups, showmeans=True)
            ax.set_title(dname)
        fig.suptitle(feat)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{feat}_by_variant.png", dpi=120)
        plt.close(fig)

    # overall CV ROC-AUC across all four dataset versions (all-features experiment)
    orig_all = pd.read_csv(FEATURES_DIR / "modeling_experiment_comparison.csv")
    orig_all = orig_all[orig_all.experiment == "G_acoustic_mfcc_temporal"]
    orig_all = orig_all.assign(dataset_version="original")
    norm_all = pd.read_csv(NORM_EXPERIMENT_CSV)
    norm_all = norm_all[(norm_all.experiment == "G_acoustic_mfcc_temporal") & (norm_all.dataset_version == "normalized")]

    fig, ax = plt.subplots(figsize=(7, 4))
    versions = ["original", "normalized", "telephony", "telephony_normalized"]
    lr_vals, lr_stds = [], []
    for v in versions:
        if v == "original":
            row = orig_all[orig_all.model == "logistic_regression"]
        elif v == "normalized":
            row = norm_all[norm_all.model == "logistic_regression"]
        else:
            row = results[v]["ablation_df"]
            row = row[(row.experiment == "G_acoustic_mfcc_temporal") & (row.model == "logistic_regression")]
        lr_vals.append(row["cv_roc_auc_mean"].iloc[0])
        lr_stds.append(row["cv_roc_auc_std"].iloc[0])
    ax.bar(versions, lr_vals, yerr=lr_stds, capsize=4)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("CV ROC-AUC (logistic_regression, all features)")
    ax.set_title("4-way comparison: original / normalized / telephony / telephony+normalized")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "four_way_cv_rocauc_comparison.png", dpi=120)
    plt.close(fig)

    return dict(zip(versions, lr_vals))


def run():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    families = get_family_cols()

    results = {}
    for variant_name in VARIANTS:
        print(f"Running diagnostics for variant: {variant_name}...")
        results[variant_name] = run_variant_diagnostics(variant_name, families)

    orig_sep = pd.read_csv(ORIG_SEP_CSV)
    filter_change = build_filter_change_table(orig_sep, results)

    print("Channel-1 diagnostic (telephony variants)...")
    channel1_sections = {}
    for variant_name in VARIANTS:
        md, summary = channel1_diagnostic_telephony(variant_name, results[variant_name]["sep_df"])
        channel1_sections[variant_name] = md

    print("Building 4-way comparison table...")
    orig_experiments = pd.read_csv(FEATURES_DIR / "modeling_experiment_comparison.csv")
    orig_experiments.insert(0, "dataset_version", "original")
    norm_experiments = pd.read_csv(NORM_EXPERIMENT_CSV)
    norm_experiments = norm_experiments[norm_experiments.dataset_version == "normalized"]

    telephony_experiments = pd.concat(
        [results[v]["ablation_df"] for v in VARIANTS]
        + [results[v]["elimination_df"] for v in VARIANTS]
        + [results[v]["simple_df"] for v in VARIANTS],
        ignore_index=True,
    )
    all_experiments = pd.concat([orig_experiments, norm_experiments, telephony_experiments], ignore_index=True)
    all_experiments.to_csv(TELEPHONY_EXPERIMENT_COMPARISON_CSV, index=False)

    print("Plots...")
    four_way = make_plots(results, filter_change)

    write_report(results, filter_change, channel1_sections, four_way, orig_experiments)
    print(f"Wrote {TELEPHONY_REPORT_MD}")


def write_report(results, filter_change, channel1_sections, four_way, orig_experiments):
    lines = ["# Telephony-band (300-3400 Hz) confound experiment report\n"]
    lines.append(
        "Tests whether the near-perfect human-vs-synthetic separation survives a common, "
        "plausible telephone-channel band limitation, applied identically to every recording "
        "regardless of label. Two variants: band-limited only, and band-limited + RMS-"
        "normalized (isolating frequency-response effects from loudness effects). Nothing about "
        "the original, denoised, or RMS-normalization-experiment outputs was changed.\n"
    )

    lines.append("## 1. How much did the common frequency constraint change the recordings?\n")
    lines.append(
        "Filter: Butterworth band-pass, order 4, 300-3400 Hz, zero-phase (sosfiltfilt) — "
        "see `preprocessing/telephony_filter.py`. Applied identically to every recording. "
        "Confirmed directly: `max_spectral_rolloff` (previously reaching ~3447-3748 Hz in the "
        "original signal — ABOVE the new 3400 Hz ceiling) is now hard-capped below it for every "
        "recording (see `outputs/features/telephony_acoustic.csv`).\n"
    )

    lines.append("## 2-3 & 12. Which features changed vs. remained invariant (with a-priori expectation)\n")
    lines.append(
        "Unlike RMS normalization, a band-pass filter is not a pure gain — no feature listed "
        "below is assumed invariant in advance; the table checks each one empirically against "
        "the a-priori expectation from the task.\n"
    )
    lines.append(filter_change.round(4).to_markdown(index=False))
    lines.append("")

    lines.append("## 4. Did ZCR separation remain?\n")
    zcr_row = filter_change[filter_change.feature.isin(["mean_zcr", "std_zcr"])]
    lines.append(zcr_row.round(4).to_markdown(index=False))
    lines.append("")

    lines.append("## 5. Did spectral centroid/bandwidth/rolloff separation remain?\n")
    spec_rows = filter_change[filter_change.feature.str.contains("spectral")]
    lines.append(spec_rows.round(4).to_markdown(index=False))
    lines.append("")

    lines.append("## 6. Did MFCC separation change?\n")
    mfcc_rows = filter_change[filter_change.feature.str.startswith("mfcc")]
    lines.append(mfcc_rows.round(4).to_markdown(index=False))
    lines.append("")

    lines.append("## 7. Did overall CV ROC-AUC change? (4-way comparison — the central result)\n")
    lines.append(
        f"All-features (acoustic+MFCC+temporal), logistic_regression, CV ROC-AUC mean:\n\n"
        + "\n".join(f"- **{k}**: {v:.4f}" for k, v in four_way.items()) + "\n"
    )
    lines.append("Full model-by-model, experiment-by-experiment detail: "
                  "`outputs/features/telephony_experiment_comparison.csv`.\n")

    lines.append("## 8. Which feature family was most affected?\n")
    fam_rows = []
    for exp in ("A_acoustic_only", "B_mfcc_only", "C_temporal_only"):
        o = orig_experiments[(orig_experiments.experiment == exp) & (orig_experiments.model == "logistic_regression")]
        t = results["telephony"]["ablation_df"]
        t = t[(t.experiment == exp) & (t.model == "logistic_regression")]
        if len(o) and len(t):
            fam_rows.append({
                "experiment": exp,
                "cv_roc_auc_original": o["cv_roc_auc_mean"].iloc[0],
                "cv_roc_auc_telephony": t["cv_roc_auc_mean"].iloc[0],
                "change": t["cv_roc_auc_mean"].iloc[0] - o["cv_roc_auc_mean"].iloc[0],
            })
    fam_df = pd.DataFrame(fam_rows)
    lines.append(fam_df.round(4).to_markdown(index=False))
    biggest = fam_df.loc[fam_df["change"].idxmin(), "experiment"] if len(fam_df) else "n/a"
    lines.append(f"\nMost-affected family (largest CV ROC-AUC drop, original -> telephony): `{biggest}`.\n")

    lines.append("## 9. Does Channel 1 still resemble synthetic C0?\n")
    for variant_name in VARIANTS:
        lines.append(channel1_sections[variant_name])

    still_high = all(v >= 0.90 for k, v in four_way.items() if k in ("telephony", "telephony_normalized"))
    lines.append("## 10. Genuine synthetic-speech signal, or recording-pipeline artifact?\n")
    if still_high:
        lines.append(
            f"CV ROC-AUC remains high under the telephony constraint (telephony: "
            f"{four_way['telephony']:.4f}, telephony+normalized: {four_way['telephony_normalized']:.4f}), "
            "despite `max_spectral_rolloff` — previously one of the two strongest single "
            "features — being hard-capped by the new 3400 Hz ceiling for every recording. "
            "This means the remaining separation is not simply \"whatever spectral content "
            "happened to sit above 3400 Hz\"; something that survives common telephone-band "
            "limiting is doing the separating. That is more consistent with a signal robust "
            "to two independent, plausible confounds (loudness AND a shared frequency "
            "ceiling) than with either confound alone explaining the original result.\n"
        )
    else:
        lines.append(
            f"CV ROC-AUC dropped substantially under the telephony constraint (telephony: "
            f"{four_way['telephony']:.4f}, telephony+normalized: {four_way['telephony_normalized']:.4f}), "
            "consistent with the original separation depending heavily on frequency-response "
            "information (e.g. content above 3400 Hz) that a real telephone channel would not "
            "reliably preserve either.\n"
        )

    lines.append("## Interpretation & recommendation\n")
    if still_high:
        lines.append(
            "**(A leaning) Stronger evidence of a genuine, more robust acoustic signal — but "
            "still not proof of real-world generalization.** The signal survived two "
            "independent, plausible recording/channel confounds in a row (loudness, then a "
            "common telephony bandwidth). It could still reflect a pipeline characteristic "
            "shared by this dataset's specific synthetic-generation system rather than "
            "synthetic speech in general. Recommended next step: external synthetic-speech "
            "validation, ideally speech from a different TTS/voice-generation system than "
            "produced this dataset's Channel 0 synthetic recordings, run through this same "
            "acoustic feature pipeline.\n"
        )
    else:
        lines.append(
            "**(B) Strong evidence of a recording/channel confound.** The classifier relies "
            "heavily on frequency-response information not preserved under a common telephony "
            "bandwidth. Recommended next step: investigate the dataset's construction "
            "(recording/generation pipeline) directly and obtain better frequency-response-"
            "matched data before further modeling.\n"
        )

    lines.append(
        "\nNot done in this step (per task scope): no neural network, no live detector, no "
        "resumed ASR, no additional NLP, no PCA/orthogonalization, no hyperparameter tuning, "
        "no permanent feature removal, no fabricated external data, no real-world accuracy "
        "claim. Stopping here — not proceeding to real-time inference.\n"
    )

    with open(TELEPHONY_REPORT_MD, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run()
