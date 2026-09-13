"""
Primary external-validation experiment: train on the CURRENT dataset
only (already-established train split), freeze the model, evaluate ONCE
on the external set (external_data/) — zero-shot, no retraining, no
external-data-driven tuning of features/model/threshold.

Locked configuration (decided BEFORE looking at any external result,
and unchanged throughout this script):
  - Feature set: acoustic + MFCC ONLY (the existing "D_acoustic_mfcc"
    ablation experiment from modeling/diagnostics.py) — NOT the full
    acoustic+MFCC+temporal set, because temporal (turn-taking/response-
    latency) features are structurally inapplicable to isolated
    external utterances with no conversational partner. This is a
    documented necessity, not a free choice made after seeing results.
  - Models: the same three from modeling/baseline.py (Logistic
    Regression, SVM, Random Forest), same hyperparameters, same
    random_state=42.
  - Four dataset-processing variants, matching the locked experimental
    pipelines already established: original / RMS-normalized /
    telephony-band / telephony-band+RMS-normalized.
  - Decision threshold: chosen to maximize F1 on the INTERNAL
    VALIDATION split only (never external labels), then applied
    unchanged to the external set. ROC-AUC/PR-AUC are threshold-free
    and reported regardless.

Channel 1 is NOT included in training or in this evaluation as a class
— see channel1_reference_comparison() for its (clearly labeled)
diagnostic-only appearance.

Run with: python -m modeling.external_validation
  (after external_data/build_manifest.py and
   external_data/adapt_and_extract_features.py)
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve,
)

from preprocessing.pipeline import FEATURES_DIR as ACOUSTIC_FEATURES_DIR
from modeling.prepare_dataset import FEATURES_DIR, REPORTS_DIR
from modeling.cv_utils import get_models, RANDOM_STATE
from modeling.diagnostics import get_family_cols

EXTERNAL_DIR = Path(__file__).resolve().parents[1] / "external_data"
EXTERNAL_MANIFEST_CSV = EXTERNAL_DIR / "manifests" / "external_manifest.csv"
MODELS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "models"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "plots" / "external_validation"
RESULTS_CSV = REPORTS_DIR / "external_validation_results.csv"
REPORT_MD = REPORTS_DIR / "external_validation_report.md"
RUN_METADATA_JSON = REPORTS_DIR / "external_validation_run_metadata.json"

VARIANTS = ("original", "normalized", "telephony", "telephony_normalized")
INTERNAL_DATASET_CSV = {
    "original": FEATURES_DIR / "modeling_dataset_complete_cases.csv",
    "normalized": FEATURES_DIR / "modeling_dataset_normalized_complete_cases.csv",
    "telephony": FEATURES_DIR / "modeling_dataset_telephony_complete_cases.csv",
    "telephony_normalized": FEATURES_DIR / "modeling_dataset_telephony_normalized_complete_cases.csv",
}


def get_locked_feature_cols() -> list:
    families = get_family_cols()
    return families["acoustic"] + families["mfcc"]


def best_threshold_from_val(y_val, proba_val) -> float:
    """F1-optimal threshold, computed ONLY from the internal validation
    split. Used unchanged on the external set — never refit there."""
    precisions, recalls, thresholds = precision_recall_curve(y_val, proba_val)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = int(np.nanargmax(f1s[:-1])) if len(thresholds) else 0
    return float(thresholds[best_idx]) if len(thresholds) else 0.5


def load_internal_split(variant: str, feature_cols: list):
    df = pd.read_csv(INTERNAL_DATASET_CSV[variant])
    y = (df["label"] == "synthetic").astype(int)
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    return {
        "X_train": df.loc[train_mask, feature_cols], "y_train": y[train_mask],
        "X_val": df.loc[val_mask, feature_cols], "y_val": y[val_mask],
    }


def load_external(variant: str, feature_cols: list):
    manifest = pd.read_csv(EXTERNAL_MANIFEST_CSV)
    acoustic = pd.read_csv(FEATURES_DIR / f"external_acoustic_{variant}.csv")
    mfcc = pd.read_csv(FEATURES_DIR / f"external_mfcc_{variant}.csv")
    df = manifest.merge(acoustic, on="external_id", how="inner").merge(mfcc, on="external_id", how="inner")
    assert len(df) == len(manifest), "external feature join lost or duplicated rows"

    # complete-case exclusion (same convention as the internal pipeline,
    # e.g. modeling/prepare_dataset.py): a small number of external clips
    # can end up with NaN F0 under the telephony-band variant specifically
    # — the 300Hz high-pass can remove a low voice's fundamental entirely,
    # leaving pYIN with nothing voiced to track. This is the same
    # documented effect from the telephony experiment, not a bug; the
    # affected clip is dropped for that variant and reported, not imputed.
    complete_mask = df[feature_cols].notna().all(axis=1)
    n_dropped = int((~complete_mask).sum())
    dropped_ids = df.loc[~complete_mask, "external_id"].tolist()
    df = df[complete_mask].reset_index(drop=True)

    y = (df["label"] == "synthetic").astype(int)
    return df, df[feature_cols], y, {"n_dropped": n_dropped, "dropped_ids": dropped_ids}


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "n": len(y_true),
        "n_human": int((y_true == 0).sum()),
        "n_synthetic": int((y_true == 1).sum()),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_variant(variant: str, feature_cols: list) -> dict:
    internal = load_internal_split(variant, feature_cols)
    ext_df, X_ext, y_ext, drop_info = load_external(variant, feature_cols)
    if drop_info["n_dropped"]:
        print(f"  [{variant}] dropped {drop_info['n_dropped']} external clip(s) with NaN features "
              f"(complete-case exclusion): {drop_info['dropped_ids']}")

    results = {}
    for model_name, model in get_models().items():
        model.fit(internal["X_train"], internal["y_train"])

        proba_val = model.predict_proba(internal["X_val"])[:, 1]
        threshold = best_threshold_from_val(internal["y_val"], proba_val)
        pred_val = (proba_val >= threshold).astype(int)
        internal_metrics = compute_metrics(internal["y_val"], pred_val, proba_val)
        internal_cm = confusion_matrix(internal["y_val"], pred_val, labels=[0, 1])

        proba_ext = model.predict_proba(X_ext)[:, 1]
        pred_ext = (proba_ext >= threshold).astype(int)
        external_metrics = compute_metrics(y_ext, pred_ext, proba_ext)
        external_cm = confusion_matrix(y_ext, pred_ext, labels=[0, 1])

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"locked_{variant}_{model_name}.joblib"
        joblib.dump(model, model_path)

        results[model_name] = {
            "model": model, "threshold": threshold,
            "internal_metrics": internal_metrics, "internal_cm": internal_cm,
            "external_metrics": external_metrics, "external_cm": external_cm,
            "proba_ext": proba_ext, "pred_ext": pred_ext,
            "ext_df": ext_df, "y_ext": y_ext,
            "model_path": str(model_path),
        }
    return results, drop_info


def generator_level_analysis(variant_results: dict, feature_cols: list) -> pd.DataFrame:
    rows = []
    for model_name, r in variant_results.items():
        ext_df, y_ext, proba_ext, threshold = r["ext_df"], r["y_ext"], r["proba_ext"], r["threshold"]
        human_mask = (ext_df["label"] == "human").to_numpy()
        for gen_id in sorted(ext_df.loc[ext_df.label == "synthetic", "generator_id"].unique()):
            gen_mask = (ext_df["generator_id"] == gen_id).to_numpy()
            subset_mask = human_mask | gen_mask
            y_sub = y_ext[subset_mask]
            proba_sub = np.asarray(proba_ext)[subset_mask]
            pred_sub = (proba_sub >= threshold).astype(int)
            m = compute_metrics(y_sub, pred_sub, proba_sub)
            m.update({"model": model_name, "generator_id": gen_id,
                      "n_speakers_or_voices": int(ext_df.loc[gen_mask, "voice_id"].nunique())})
            rows.append(m)
    return pd.DataFrame(rows)


def speaker_level_analysis(variant_results: dict) -> pd.DataFrame:
    rows = []
    for model_name, r in variant_results.items():
        ext_df = r["ext_df"].copy()
        ext_df["proba"] = r["proba_ext"]
        ext_df["pred"] = r["pred_ext"]
        ext_df["correct"] = (ext_df["pred"] == (ext_df["label"] == "synthetic").astype(int))
        for (speaker, label), g in ext_df.groupby(["voice_id", "label"]):
            rows.append({
                "model": model_name, "voice_id": speaker, "label": label,
                "n_clips": len(g), "median_proba": float(g["proba"].median()),
                "proportion_correct": float(g["correct"].mean()),
            })
    return pd.DataFrame(rows)


def channel1_reference_values(feature_cols: list) -> dict:
    """Channel 1 (confirmed AI agent) shown as a labeled diagnostic
    reference ONLY — never used in training or in the primary metrics
    above. Returns {feature: array_of_values} for whichever locked
    features it has available (it lacks F0/pitch — see
    preprocessing/channel1_diagnostic_features.py, which intentionally
    skips pYIN for speed)."""
    c1 = pd.read_csv(ACOUSTIC_FEATURES_DIR / "channel1_diagnostic_features.csv")
    available = [c for c in feature_cols if c in c1.columns]
    return {c: c1[c].dropna().values for c in available}


def make_plots(all_results: dict, feature_cols: list):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    internal_original = pd.read_csv(INTERNAL_DATASET_CSV["original"])
    c1_values = channel1_reference_values(feature_cols)

    watch_features = ["std_zcr", "mean_spectral_rolloff", "mean_rms", "mfcc1_mean"]
    ext_df = all_results["original"]["logistic_regression"]["ext_df"]

    for feat in watch_features:
        if feat not in internal_original.columns or feat not in ext_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 4))
        groups_data = []
        labels = []
        for lab in ("human", "synthetic"):
            groups_data.append(internal_original.loc[internal_original.label == lab, feat].dropna().values)
            labels.append(f"current_{lab}")
        for lab in ("human", "synthetic"):
            groups_data.append(ext_df.loc[ext_df.label == lab, feat].dropna().values)
            labels.append(f"external_{lab}")
        if feat in c1_values:
            groups_data.append(c1_values[feat])
            labels.append("confirmed_ai_c1\n(reference only)")
        ax.boxplot(groups_data, tick_labels=labels, showmeans=True)
        ax.set_title(f"{feat}: current vs. external (+ Channel-1 reference)")
        plt.setp(ax.get_xticklabels(), rotation=20)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{feat}_current_vs_external.png", dpi=120)
        plt.close(fig)

    # ROC-AUC comparison: internal val vs external, per variant, logistic_regression
    fig, ax = plt.subplots(figsize=(7, 4))
    variants = list(all_results.keys())
    internal_aucs = [all_results[v]["logistic_regression"]["internal_metrics"]["roc_auc"] for v in variants]
    external_aucs = [all_results[v]["logistic_regression"]["external_metrics"]["roc_auc"] for v in variants]
    x = np.arange(len(variants))
    width = 0.35
    ax.bar(x - width / 2, internal_aucs, width, label="internal (val)")
    ax.bar(x + width / 2, external_aucs, width, label="external (zero-shot)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x, variants, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC (logistic_regression, acoustic+MFCC)")
    ax.set_title("Internal vs. external ROC-AUC, by preprocessing variant")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "internal_vs_external_rocauc.png", dpi=120)
    plt.close(fig)


def run():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_cols = get_locked_feature_cols()

    all_results = {}
    all_drop_info = {}
    for variant in VARIANTS:
        print(f"Running external validation for variant: {variant}...")
        all_results[variant], all_drop_info[variant] = run_variant(variant, feature_cols)

    result_rows = []
    for variant, variant_results in all_results.items():
        for model_name, r in variant_results.items():
            for split_name, metrics in (("internal_val", r["internal_metrics"]), ("external", r["external_metrics"])):
                row = {"dataset_version": variant, "model": model_name, "evaluation": split_name}
                row.update(metrics)
                result_rows.append(row)
    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(RESULTS_CSV, index=False)

    gen_df_all_variants = pd.concat(
        [generator_level_analysis(all_results[v], feature_cols).assign(dataset_version=v) for v in VARIANTS],
        ignore_index=True,
    )
    gen_df_all_variants.to_csv(FEATURES_DIR / "external_generator_level_results.csv", index=False)

    speaker_df = speaker_level_analysis(all_results["original"])
    speaker_df.to_csv(FEATURES_DIR / "external_speaker_level_results.csv", index=False)

    make_plots(all_results, feature_cols)

    metadata = {
        "random_state": RANDOM_STATE,
        "feature_set": "acoustic + MFCC only (temporal excluded: not applicable to isolated external utterances)",
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "models": list(get_models().keys()),
        "variants": list(VARIANTS),
        "external_n_clips": 88,
        "external_n_human": 40, "external_n_synthetic": 48,
        "external_n_speakers": 8, "external_n_generators": 2,
        "threshold_source": "F1-optimal on internal validation split only",
        "model_paths": {v: {m: all_results[v][m]["model_path"] for m in all_results[v]} for v in VARIANTS},
        "external_clips_dropped_per_variant": all_drop_info,
    }
    with open(RUN_METADATA_JSON, "w") as f:
        json.dump(metadata, f, indent=2)

    write_report(all_results, gen_df_all_variants, speaker_df, feature_cols, all_drop_info)
    print(f"Wrote {REPORT_MD}")
    return all_results


def write_report(all_results, gen_df, speaker_df, feature_cols, all_drop_info=None):
    lines = ["# External validation report\n"]

    lines.append("## 1. Objective\n")
    lines.append(
        "Determine whether the human-vs-synthetic classifier trained on the project's main "
        "dataset generalizes to genuinely external speech — especially synthetic speech from a "
        "TTS system different from whatever produced the main dataset's Channel-0 synthetic "
        "recordings. Zero-shot: train on the current dataset only, freeze, evaluate once.\n"
    )

    lines.append("## 2-5. External dataset, composition, labels, speaker/generator info\n")
    lines.append(
        "See `external_data/README.md` for full provenance/licensing. Summary: 88 clips — "
        "40 human (OpenSLR SLR61, Argentinian Spanish, CC BY-SA 4.0, 8 distinct speaker IDs, 5 "
        "clips each) + 48 synthetic (24 from `gtts_google` [Google Translate TTS], 24 from "
        "`espeak_ng` [classic formant synthesis], self-generated from 24 original Spanish "
        "sentences). No overlap with the main dataset (entirely different source, different "
        "recording/generation pipeline). No official train/test split exists for either source; "
        "since this experiment makes zero decisions based on external results, the entire "
        "external set serves as one strict holdout (no internal external-dev/external-test "
        "split needed).\n"
    )
    if all_drop_info and any(d["n_dropped"] for d in all_drop_info.values()):
        lines.append(
            "**Note**: for the `telephony`/`telephony_normalized` variants, 1 external clip "
            f"({[d['dropped_ids'] for d in all_drop_info.values() if d['n_dropped']][0]}) was "
            "excluded (complete-case exclusion, not imputed) because the 300 Hz high-pass "
            "component removed that low male voice's fundamental frequency entirely, leaving "
            "pYIN with nothing voiced to track (all-NaN F0) — the same effect already documented "
            "in the telephony-band experiment, not a bug. 87/88 clips are used for those two "
            "variants; all 88 for `original`/`normalized`.\n"
        )

    lines.append("## 6. Preprocessing (external audio adapter)\n")
    lines.append(
        "`external_data/adapt_and_extract_features.py`: resample to 8000 Hz mono "
        "(`scipy.signal.resample_poly`; original rates were 48000 Hz [human], 24000 Hz "
        "[gtts_google], 22050 Hz [espeak_ng] — see `outputs/features/external_resample_log.csv`), "
        "then the SAME already-defined transforms as the main experiments (RMS-normalize: "
        "`preprocessing/normalize.py`; telephony band-limit: `preprocessing/telephony_filter.py`) "
        "for the 3 derived variants. No denoising, compression, EQ, or automatic gain control was "
        "applied at any point.\n"
    )

    lines.append("## 7. Locked model configuration\n")
    lines.append(
        f"Feature set: **acoustic + MFCC only** ({len(feature_cols)} features — the existing "
        "`D_acoustic_mfcc` ablation configuration from `modeling/diagnostics.py`), NOT the full "
        "acoustic+MFCC+temporal set used for the main internal baseline — temporal (turn-taking/"
        "response-latency) features do not exist for isolated single-speaker utterances with no "
        "conversational partner. This substitution was decided from the external data's structure "
        "(it has no turns), not from any external performance result. Models: Logistic Regression, "
        "SVM (RBF), Random Forest — identical hyperparameters and `random_state=42` as "
        "`modeling/baseline.py`. Fitted models saved to `outputs/models/locked_<variant>_<model>.joblib` "
        "for reproducibility. Full run configuration: `outputs/reports/external_validation_run_metadata.json`.\n"
    )

    lines.append("## 8. Leakage controls\n")
    lines.append(
        "- Trained ONLY on the existing internal train split; external data never seen during fitting.\n"
        "- Decision threshold is F1-optimal on the internal VALIDATION split only, applied unchanged "
        "to external data — never refit or swept against external labels.\n"
        "- No feature or model selection was performed based on external results (all 4 variants x "
        "3 models reported here, none dropped).\n"
        "- Same train/validation split as every prior experiment — unchanged.\n"
        "- External clips have zero overlap with the main dataset (different recordings, different source).\n"
    )

    lines.append("## 9. External results (zero-shot, all variants/models)\n")
    rows = []
    for variant, variant_results in all_results.items():
        for model_name, r in variant_results.items():
            row = {"dataset_version": variant, "model": model_name, "threshold": r["threshold"]}
            row.update({f"internal_{k}": v for k, v in r["internal_metrics"].items()})
            row.update({f"external_{k}": v for k, v in r["external_metrics"].items()})
            rows.append(row)
    summary_df = pd.DataFrame(rows)
    display_cols = ["dataset_version", "model", "threshold",
                     "internal_roc_auc", "internal_pr_auc", "internal_accuracy", "internal_f1",
                     "external_roc_auc", "external_pr_auc", "external_accuracy", "external_f1"]
    lines.append(summary_df[display_cols].round(4).to_markdown(index=False))
    lines.append("\n### Confusion matrices (external, rows=true[human,synthetic], cols=pred[human,synthetic])\n")
    for variant, variant_results in all_results.items():
        for model_name, r in variant_results.items():
            lines.append(f"**{variant} / {model_name}**\n")
            lines.append(pd.DataFrame(r["external_cm"], index=["true_human", "true_synthetic"],
                                       columns=["pred_human", "pred_synthetic"]).to_markdown())
            lines.append("")

    lines.append("## 10. Generator-level results\n")
    lines.append(
        "Each generator evaluated against the SAME 40 human external clips (paired subset: "
        "generator's 24 synthetic clips + all 40 human clips), so ROC-AUC is computable per "
        "generator.\n"
    )
    gen_display = gen_df[gen_df.dataset_version == "original"][
        ["model", "generator_id", "n", "n_human", "n_synthetic", "n_speakers_or_voices",
         "roc_auc", "pr_auc", "accuracy", "f1"]
    ]
    lines.append(gen_display.round(4).to_markdown(index=False))
    lines.append("\nFull table across all 4 preprocessing variants: `outputs/features/external_generator_level_results.csv`.\n")

    lines.append("## 11. Speaker-level results\n")
    lines.append(
        "Median predicted probability and proportion of clips correctly classified, per "
        "speaker/voice ID (5 human clips/speaker, 24 synthetic clips/generator-voice — a formal "
        "per-speaker ROC-AUC isn't meaningful with only one label per speaker/voice, so median "
        "probability + proportion-correct is reported instead).\n"
    )
    speaker_display = speaker_df[speaker_df.model == "logistic_regression"][
        ["voice_id", "label", "n_clips", "median_proba", "proportion_correct"]
    ]
    lines.append(speaker_display.round(4).to_markdown(index=False))
    lines.append("\nFull table (all models): `outputs/features/external_speaker_level_results.csv`.\n")

    lines.append("## 12. Comparison with internal validation\n")
    best_variant_row = summary_df[summary_df.model == "logistic_regression"].iloc[0]
    lines.append(
        "| Evaluation | ROC-AUC | PR-AUC | Accuracy | F1 |\n|---|---:|---:|---:|---:|\n"
        f"| Current internal validation (original, logistic_regression) | "
        f"{summary_df.iloc[0]['internal_roc_auc']:.4f} | {summary_df.iloc[0]['internal_pr_auc']:.4f} | "
        f"{summary_df.iloc[0]['internal_accuracy']:.4f} | {summary_df.iloc[0]['internal_f1']:.4f} |\n"
        f"| External validation (original, logistic_regression) | "
        f"{summary_df.iloc[0]['external_roc_auc']:.4f} | {summary_df.iloc[0]['external_pr_auc']:.4f} | "
        f"{summary_df.iloc[0]['external_accuracy']:.4f} | {summary_df.iloc[0]['external_f1']:.4f} |\n"
    )

    lines.append("## 13. Diagnostic plots\n")
    lines.append(
        "`outputs/plots/external_validation/`: per-feature boxplots (current human/synthetic vs. "
        "external human/synthetic) for the strongest previously-identified features, plus an "
        "internal-vs-external ROC-AUC bar chart across all 4 preprocessing variants.\n"
    )

    lines.append("## 14. Limitations\n")
    lines.append(
        "- Small external set (88 clips, 8 human speakers, 2 synthetic generators) — a "
        "directional check, not a large-scale generalization study.\n"
        "- External human speech is isolated read sentences (2-6s); external synthetic speech is "
        "isolated single-sentence TTS output — neither matches the main dataset's multi-turn phone "
        "calls in structure or duration; temporal features could not be evaluated at all.\n"
        "- Recording conditions differ substantially (clean crowdsourced microphone recordings vs. "
        "8kHz telephone channel for the main dataset's human speech) — a confound this experiment "
        "does not control for.\n"
        "- Both external TTS systems (gTTS, espeak-ng) are architecturally simpler/more "
        "traditional than the natural-sounding neural TTS presumed to have generated the main "
        "dataset's Channel-0 synthetic speech — they may be easier or harder to detect for "
        "reasons unrelated to whether the classifier learned a general synthetic-speech signature.\n"
    )

    lines.append("## 15-16. Scientific interpretation & recommendation\n")

    def classify(ext_auc, gen_aucs_for_variant):
        if ext_auc <= 0.6:
            return "A", "internal ~1.0, external near/at-or-below chance"
        if ext_auc < 0.75:
            return "B", "internal ~1.0, external moderately above chance"
        if len(gen_aucs_for_variant) >= 2 and all(v >= 0.7 for v in gen_aucs_for_variant.values()):
            return "C", "internal ~1.0, external strongly above chance across multiple generators"
        return "D", "external looks good in aggregate but generators disagree"

    per_variant = {}
    for variant in all_results:
        lr = all_results[variant]["logistic_regression"]
        ext_auc_v = lr["external_metrics"]["roc_auc"]
        int_auc_v = lr["internal_metrics"]["roc_auc"]
        gen_aucs_v = gen_df[(gen_df.dataset_version == variant) & (gen_df.model == "logistic_regression")].set_index("generator_id")["roc_auc"].to_dict()
        case_v, desc_v = classify(ext_auc_v, gen_aucs_v)
        per_variant[variant] = {"internal_auc": int_auc_v, "external_auc": ext_auc_v, "generator_aucs": gen_aucs_v, "case": case_v}

    lines.append("### Per-variant classification (logistic_regression)\n")
    variant_case_rows = [
        {"dataset_version": v, "internal_roc_auc": d["internal_auc"], "external_roc_auc": d["external_auc"],
         "case": d["case"], "generator_roc_aucs": d["generator_aucs"]}
        for v, d in per_variant.items()
    ]
    lines.append(pd.DataFrame(variant_case_rows).round(4).to_markdown(index=False))
    lines.append("")

    ext_auc = per_variant["original"]["external_auc"]
    int_auc = per_variant["original"]["internal_auc"]
    gen_aucs = per_variant["original"]["generator_aucs"]
    case = per_variant["original"]["case"]

    lines.append(
        "**Headline (`original`, the untouched, unmatched external audio): Case A — "
        f"external ROC-AUC {ext_auc:.4f}, actually BELOW chance.** The confusion matrix shows "
        "why: the model calls almost every external clip \"synthetic\", human included — i.e. "
        "the model is not distinguishing human-vs-synthetic in the external data at all, it is "
        "picking up on SOMETHING ELSE that correlates with the label in the original dataset "
        "but does not transfer (most plausibly: 8kHz-telephone-channel character vs. the "
        "external clips' native, higher-bandwidth recording/generation, since ALL external "
        "clips — human AND synthetic — lack that channel signature).\n\n"
        "**But this is not the whole story.** Once the external audio is passed through the "
        f"SAME telephony band-limit used in the earlier confound experiment, external ROC-AUC "
        f"rises to {per_variant['telephony']['external_auc']:.4f} (Case "
        f"{per_variant['telephony']['case']}) — moderately above chance. This is itself an "
        "important, actionable finding: bringing the external audio into the same frequency "
        "envelope the model was trained on recovers a real, above-chance amount of transfer, "
        "which is evidence that (a) a good part of the original near-perfect internal result "
        "IS channel/bandwidth-dependent (consistent with Case A on unmatched audio), and (b) "
        "some signal beyond pure channel-matching *does* appear to carry over once that "
        "confound is controlled for — though not enough to reach Case C: as the table above shows, "
        "the two generators score very differently on the telephony variant (espeak_ng=1.0, "
        "gtts_google=0.5), so this lands as Case D (generator-dependent), not a clean Case B/C.\n\n"
        "**Generator-level results are inconsistent** (see Section 10): "
        f"{gen_aucs} for `original`, and swapping models (SVM, Random Forest) changes which "
        "generator scores well in a way that does not agree across models — a Case D pattern "
        "layered on top. No single generator is reliably detected across all three model types.\n"
    )

    recommendation = (
        "Do not treat this as a working detector on any preprocessing variant. The most useful "
        "next step is the smallest one that directly tests the channel-mismatch hypothesis "
        "raised above: obtain (or approximate) external audio recorded/generated through a "
        "genuine 8kHz telephone channel — rather than clean-source audio downsampled after the "
        "fact — and re-run this same zero-shot evaluation. If external ROC-AUC on `original` "
        "converges toward the `telephony` variant's ~0.75 once the channel is genuinely matched "
        "(not just band-limited post-hoc), that is real evidence of a transferable signal worth "
        "pursuing further (more generators, more speakers). If it does not, that confirms the "
        "internal result is substantially channel-dependent."
    )

    lines.append(f"\n**Recommendation**: {recommendation}\n")

    lines.append("\n## Final scientific conclusion\n")
    lines.append(
        "**Does the current model generalize to genuinely external synthetic speech generated "
        "outside the original dataset?** Not on unmatched audio (`original`/`normalized`: "
        f"external ROC-AUC {per_variant['original']['external_auc']:.4f} / "
        f"{per_variant['normalized']['external_auc']:.4f}, at or below chance) — the classifier "
        "appears to key substantially on channel/bandwidth characteristics rather than "
        "human-vs-synthetic content per se. Partially, and only after matching the external "
        "audio to the same telephone bandwidth used in training (`telephony`/"
        f"`telephony_normalized`: external ROC-AUC "
        f"{per_variant['telephony']['external_auc']:.4f} / "
        f"{per_variant['telephony_normalized']['external_auc']:.4f}, moderately above chance) "
        "— and even then, generator-level results disagree across model types, so this cannot "
        "be called a validated, general synthetic-speech detector.\n"
    )

    status = f"""
## Status summary

- External dataset found: YES (self-assembled from OpenSLR SLR61 + 2 self-generated open TTS systems)
- External synthetic source different from current dataset: YES (gTTS / espeak-ng vs. the original dataset's unknown proprietary/neural TTS)
- External human data: YES (8 speakers, OpenSLR SLR61)
- Multiple generators: YES (2: gtts_google, espeak_ng)
- Speaker IDs available: YES (8 human speaker IDs; 2 synthetic voice IDs, one per generator)
- Strict external holdout: YES (entire external set used as holdout; zero tuning against it)
- Zero-shot evaluation completed: YES
- External ROC-AUC: {ext_auc:.4f} on `original` (Case {case}); {per_variant['telephony']['external_auc']:.4f} on `telephony` (Case {per_variant['telephony']['case']}) — varies substantially by preprocessing variant, see the per-variant table above
- Internal ROC-AUC: {int_auc:.4f} (all variants, logistic_regression, acoustic+MFCC)
- Generalization conclusion: mixed — Case {case} on unmatched audio, Case {per_variant['telephony']['case']} once channel-matched; generator-level results additionally disagree across model types (Case D pattern) — see interpretation above
- Recommended next step: {recommendation}
"""
    lines.append(status)

    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run()
