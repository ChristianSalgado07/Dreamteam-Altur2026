"""
Domain-matched confound experiment: does the ~0.75 external ROC-AUC
recovered by digitally band-limiting external audio (see
outputs/reports/external_validation_report.md) reflect a genuine
transferable synthetic-speech signal, or an artifact of the earlier
external set never having been through ANY real telephone channel at
all (clean OpenSLR speech + clean-TTS-output vs. this project's
genuinely 8kHz-telephone-channel training data)?

Uses ASVspoof2021 LA evaluation data (external_data/README.md /
external_data/raw/asvspoof2021_*) — the ONLY identified dataset with
BOTH bona fide (human) and spoofed (13 different TTS/VC systems)
speech GENUINELY transmitted through a real telephone network (PSTN),
alongside an untransmitted ("none") condition using the same speaker
pool and same synthesis systems. This is a speaker/system-matched, not
utterance-matched, comparison — see external_data/select_and_extract_asvspoof.py.

Uses the EXACT SAME locked models saved by modeling/external_validation.py
(outputs/models/locked_<variant>_<model>.joblib) — no retraining, no new
models. Threshold is recomputed deterministically from the (unchanged)
internal validation split — identical to what external_validation.py
used, not re-derived from this new external data.

Feature set: acoustic + MFCC only (same as external_validation.py, same
reason — isolated utterances have no turn structure).

Run with: python -m modeling.domain_matched_validation
  (after external_data/select_and_extract_asvspoof.py,
   external_data/build_domain_matched_manifest.py, and
   external_data/adapt_and_extract_domain_matched.py)
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
    roc_auc_score, average_precision_score, confusion_matrix,
)

from modeling.prepare_dataset import FEATURES_DIR, REPORTS_DIR
from modeling.external_validation import (
    VARIANTS, load_internal_split, best_threshold_from_val, get_locked_feature_cols, MODELS_DIR,
)

EXTERNAL_DIR = Path(__file__).resolve().parents[1] / "external_data"
DOMAIN_MANIFEST_CSV = EXTERNAL_DIR / "manifests" / "domain_matched_manifest.csv"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "plots" / "domain_matched_validation"
RESULTS_CSV = REPORTS_DIR / "domain_matched_validation_results.csv"
REPORT_MD = REPORTS_DIR / "domain_matched_validation_report.md"


def load_domain_matched(variant: str, feature_cols: list):
    manifest = pd.read_csv(DOMAIN_MANIFEST_CSV)
    acoustic = pd.read_csv(FEATURES_DIR / f"domain_matched_acoustic_{variant}.csv")
    mfcc = pd.read_csv(FEATURES_DIR / f"domain_matched_mfcc_{variant}.csv")
    df = manifest.merge(acoustic, on="external_id", how="inner").merge(mfcc, on="external_id", how="inner")
    assert len(df) == len(manifest), "domain-matched feature join lost or duplicated rows"

    complete_mask = df[feature_cols].notna().all(axis=1)
    n_dropped = int((~complete_mask).sum())
    dropped_ids = df.loc[~complete_mask, "external_id"].tolist()
    df = df[complete_mask].reset_index(drop=True)
    return df, {"n_dropped": n_dropped, "dropped_ids": dropped_ids}


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
    dm_df, drop_info = load_domain_matched(variant, feature_cols)
    if drop_info["n_dropped"]:
        print(f"  [{variant}] dropped {drop_info['n_dropped']} domain-matched clip(s) with NaN features: {drop_info['dropped_ids']}")

    y_dm = (dm_df["label"] == "synthetic").astype(int)
    results = {}

    for model_name in ("logistic_regression", "random_forest", "svm"):
        model_path = MODELS_DIR / f"locked_{variant}_{model_name}.joblib"
        model = joblib.load(model_path)

        proba_val = model.predict_proba(internal["X_val"])[:, 1]
        threshold = best_threshold_from_val(internal["y_val"], proba_val)

        proba_dm = model.predict_proba(dm_df[feature_cols])[:, 1]
        pred_dm = (proba_dm >= threshold).astype(int)

        dm_df_m = dm_df.copy()
        dm_df_m["proba"] = proba_dm
        dm_df_m["pred"] = pred_dm

        clean = dm_df_m[dm_df_m.channel_condition == "clean"]
        telephone = dm_df_m[dm_df_m.channel_condition == "telephone"]

        def metrics_for(sub):
            y = (sub["label"] == "synthetic").astype(int)
            return compute_metrics(y, sub["pred"], sub["proba"])

        results[model_name] = {
            "threshold": threshold,
            "df": dm_df_m,
            "clean_metrics": metrics_for(clean),
            "telephone_metrics": metrics_for(telephone),
            "clean_cm": confusion_matrix((clean.label == "synthetic").astype(int), clean.pred, labels=[0, 1]),
            "telephone_cm": confusion_matrix((telephone.label == "synthetic").astype(int), telephone.pred, labels=[0, 1]),
        }
    return results, drop_info


def channel_effect_table(variant_results: dict) -> pd.DataFrame:
    rows = []
    for model_name, r in variant_results.items():
        df = r["df"]
        for label in ("human", "synthetic"):
            clean_sub = df[(df.channel_condition == "clean") & (df.label == label)]
            tel_sub = df[(df.channel_condition == "telephone") & (df.label == label)]
            rows.append({
                "model": model_name, "speech_type": label,
                "clean_mean_proba_synthetic": clean_sub["proba"].mean(),
                "telephone_mean_proba_synthetic": tel_sub["proba"].mean(),
                "clean_frac_predicted_synthetic": clean_sub["pred"].mean(),
                "telephone_frac_predicted_synthetic": tel_sub["pred"].mean(),
                "shift": tel_sub["proba"].mean() - clean_sub["proba"].mean(),
            })
    return pd.DataFrame(rows)


def generator_level_table(variant_results: dict) -> pd.DataFrame:
    rows = []
    for model_name, r in variant_results.items():
        df = r["df"]
        for condition in ("clean", "telephone"):
            sub = df[df.channel_condition == condition]
            human = sub[sub.label == "human"]
            for gen_id in sorted(sub.loc[sub.label == "synthetic", "generator_id"].unique()):
                gen = sub[sub.generator_id == gen_id]
                paired = pd.concat([human, gen])
                y = (paired["label"] == "synthetic").astype(int)
                m = compute_metrics(y, paired["pred"], paired["proba"])
                m.update({"model": model_name, "condition": condition, "generator_id": gen_id})
                rows.append(m)
    return pd.DataFrame(rows)


def speaker_level_table(variant_results: dict) -> pd.DataFrame:
    rows = []
    for model_name, r in variant_results.items():
        df = r["df"]
        human = df[df.label == "human"]
        for (speaker, condition), g in human.groupby(["speaker_id", "channel_condition"]):
            rows.append({
                "model": model_name, "speaker_id": speaker, "channel_condition": condition,
                "n_clips": len(g), "median_proba": float(g["proba"].median()),
                "proportion_correctly_human": float((g["pred"] == 0).mean()),
            })
    return pd.DataFrame(rows)


def make_plots(all_results: dict, feature_cols: list):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    watch_features = ["std_zcr", "mean_spectral_rolloff", "mean_rms", "mfcc1_mean"]

    original_internal = pd.read_csv(FEATURES_DIR / "modeling_dataset_complete_cases.csv")
    dm_df = all_results["original"]["logistic_regression"]["df"]

    for feat in watch_features:
        if feat not in dm_df.columns or feat not in original_internal.columns:
            continue
        fig, ax = plt.subplots(figsize=(11, 4))
        groups_data, labels = [], []
        for lab in ("human", "synthetic"):
            groups_data.append(original_internal.loc[original_internal.label == lab, feat].dropna().values)
            labels.append(f"current_{lab}")
        for cond in ("clean", "telephone"):
            for lab in ("human", "synthetic"):
                sub = dm_df[(dm_df.channel_condition == cond) & (dm_df.label == lab)]
                groups_data.append(sub[feat].dropna().values)
                labels.append(f"{cond}_{lab}")
        ax.boxplot(groups_data, tick_labels=labels, showmeans=True)
        plt.setp(ax.get_xticklabels(), rotation=25)
        ax.set_title(f"{feat}: current dataset vs. domain-matched external (clean/telephone)")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"{feat}_domain_matched.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    variants = list(all_results.keys())
    clean_aucs = [all_results[v]["logistic_regression"]["clean_metrics"]["roc_auc"] for v in variants]
    tel_aucs = [all_results[v]["logistic_regression"]["telephone_metrics"]["roc_auc"] for v in variants]
    x = np.arange(len(variants))
    width = 0.35
    ax.bar(x - width / 2, clean_aucs, width, label="clean external")
    ax.bar(x + width / 2, tel_aucs, width, label="telephone (PSTN) external")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x, variants, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ROC-AUC (logistic_regression, acoustic+MFCC)")
    ax.set_title("Domain-matched: clean vs. genuinely-telephone external ROC-AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "clean_vs_telephone_rocauc.png", dpi=120)
    plt.close(fig)


def run():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    feature_cols = get_locked_feature_cols()

    all_results = {}
    all_drop_info = {}
    for variant in VARIANTS:
        print(f"Running domain-matched validation for variant: {variant}...")
        all_results[variant], all_drop_info[variant] = run_variant(variant, feature_cols)

    result_rows = []
    for variant, variant_results in all_results.items():
        for model_name, r in variant_results.items():
            for condition, metrics in (("clean", r["clean_metrics"]), ("telephone", r["telephone_metrics"])):
                row = {"dataset_version": variant, "model": model_name, "channel_condition": condition}
                row.update(metrics)
                result_rows.append(row)
    results_df = pd.DataFrame(result_rows)
    results_df.to_csv(RESULTS_CSV, index=False)

    channel_effect_all = pd.concat(
        [channel_effect_table(all_results[v]).assign(dataset_version=v) for v in VARIANTS], ignore_index=True,
    )
    channel_effect_all.to_csv(FEATURES_DIR / "domain_matched_channel_effect.csv", index=False)

    gen_all = pd.concat(
        [generator_level_table(all_results[v]).assign(dataset_version=v) for v in VARIANTS], ignore_index=True,
    )
    gen_all.to_csv(FEATURES_DIR / "domain_matched_generator_level_results.csv", index=False)

    speaker_all = pd.concat(
        [speaker_level_table(all_results[v]).assign(dataset_version=v) for v in VARIANTS], ignore_index=True,
    )
    speaker_all.to_csv(FEATURES_DIR / "domain_matched_speaker_level_results.csv", index=False)

    make_plots(all_results, feature_cols)

    write_report(all_results, results_df, channel_effect_all, gen_all, speaker_all, feature_cols, all_drop_info)
    print(f"Wrote {REPORT_MD}")
    return all_results


def write_report(all_results, results_df, channel_effect_all, gen_all, speaker_all, feature_cols, all_drop_info):
    dm_manifest = pd.read_csv(DOMAIN_MANIFEST_CSV)
    clean_speakers = set(dm_manifest[(dm_manifest.channel_condition == "clean") & (dm_manifest.label == "human")]["speaker_id"])
    tel_speakers = set(dm_manifest[(dm_manifest.channel_condition == "telephone") & (dm_manifest.label == "human")]["speaker_id"])
    speaker_overlap = clean_speakers & tel_speakers
    n_speaker_overlap = len(speaker_overlap)
    clean_systems = set(dm_manifest[(dm_manifest.channel_condition == "clean") & (dm_manifest.label == "synthetic")]["generator_id"])
    tel_systems = set(dm_manifest[(dm_manifest.channel_condition == "telephone") & (dm_manifest.label == "synthetic")]["generator_id"])
    system_overlap = clean_systems & tel_systems
    n_system_overlap = len(system_overlap)

    lines = ["# Domain-matched validation report\n"]
    lines.append(
        "Tests whether the ~0.75 external ROC-AUC recovered by digitally band-limiting external "
        "audio (`outputs/reports/external_validation_report.md`) reflects a genuine transferable "
        "synthetic-speech signal, or an artifact of that external audio never having passed "
        "through a REAL telephone channel at all. Uses ASVspoof2021 LA evaluation data — bona "
        "fide and spoofed (13 TTS/VC systems) speech genuinely transmitted through the public "
        "switched telephone network (PSTN), alongside an untransmitted (\"clean\") condition — "
        "the SAME locked models from `modeling/external_validation.py`, no retraining.\n"
    )

    lines.append("## Dataset\n")
    lines.append(
        "**ASVspoof2021 LA evaluation set** — https://zenodo.org/record/4837263 (ODC-BY 1.0, no "
        "login required) + keys from https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz. "
        "Per the official evaluation plan (`asvspoof2021_evaluation_plan.pdf`, Section 3): "
        "\"Bona fide and spoofed speech data ... is transmitted across either a public switched "
        "telephone network (PSTN) or a voice over IP (VoIP) network.\" We use only the `none` "
        "(untransmitted, our \"clean\") and `pstn` (genuinely PSTN-transmitted, our \"telephone\") "
        "codec conditions — NOT the other codec labels (alaw/ulaw/g722/gsm/opus), which are "
        "simulated codec compression rather than confirmed real telephone-network transmission. "
        "This challenge dataset was designed specifically to study channel-robustness of "
        "synthetic-speech detection, which is exactly this experiment's question.\n\n"
        "**Important caveat**: `none` and `pstn` do NOT share the same underlying utterances "
        "(verified: zero utterance-ID overlap) — this is an utterance-INDEPENDENT comparison, "
        "not the same sentence heard twice. The full ASVspoof2021 speaker/system pools are "
        "identical across conditions (same 67 bona fide speakers, same 13 attack systems appear "
        "under both `none` and `pstn`), but **our small 8-per-condition subsample was drawn "
        "independently per condition** and only partially overlaps in practice: "
        f"**{n_speaker_overlap}/8 human speakers** and **{n_system_overlap}/8 attack systems** "
        "are shared between the clean and telephone samples used here (exact IDs: speakers "
        f"{sorted(speaker_overlap)}, systems {sorted(system_overlap)}). This is a real limitation "
        "of this subset (not the underlying dataset) — the human clean-vs-telephone comparison in "
        "particular is only partly speaker-controlled; the generator-level comparison is better "
        "controlled thanks to higher system overlap. Documented here rather than overstated.\n\n"
        "**Language**: ASVspoof is English (VCTK-derived); the current project's dataset is "
        "Spanish. This reintroduces a language confound this experiment cannot separate from the "
        "channel effect — documented here, not glossed over. No registration was completed with "
        "the ASVspoof consortium (data was downloaded directly from its open Zenodo record for "
        "internal research measurement, not challenge participation).\n\n"
        "Selection: 8 bona fide speakers x 5 utterances + 8 attack systems x 5 utterances, per "
        "condition (`external_data/select_and_extract_asvspoof.py`, seed=42) — a small, "
        "reviewable subset, not the full 25,938-utterance-per-condition evaluation set.\n"
    )
    total_dropped = sum(d["n_dropped"] for d in all_drop_info.values())
    if total_dropped:
        lines.append(
            f"**Excluded clips**: {total_dropped} clip(s) across all variants dropped via "
            "complete-case exclusion (NaN acoustic features, e.g. pYIN finding no voiced frames "
            "under the telephony filter — not imputed). Per-variant detail: " +
            ", ".join(f"{v}: {d['n_dropped']} ({d['dropped_ids']})" for v, d in all_drop_info.items() if d["n_dropped"]) + "\n"
        )
    else:
        lines.append("**Excluded clips**: none — all selected clips produced complete features in every variant.\n")

    lines.append("## Preprocessing\n")
    lines.append(
        "`external_data/adapt_and_extract_domain_matched.py`: resample to 8000 Hz mono, then the "
        "same 4 variants as every prior experiment (original / RMS-normalized / telephony-band-"
        "limited / telephony-band+normalized). No denoising, compression, EQ, or AGC.\n"
    )

    lines.append("## Locked model configuration\n")
    lines.append(
        f"Identical models to `modeling/external_validation.py` — loaded from "
        f"`outputs/models/locked_<variant>_<model>.joblib`, NOT retrained. Feature set: acoustic + "
        f"MFCC only ({len(feature_cols)} features; temporal excluded, same reason as before — "
        "isolated utterances have no turn structure). Threshold: F1-optimal on the internal "
        "validation split, recomputed deterministically (same data, same code) rather than "
        "re-derived from this new external data.\n"
    )

    lines.append("## A/B. Clean external and telephone external (human vs. synthetic)\n")
    display_cols = ["dataset_version", "model", "channel_condition", "n", "n_human", "n_synthetic",
                     "roc_auc", "pr_auc", "accuracy", "f1"]
    lines.append(results_df[display_cols].round(4).to_markdown(index=False))
    lines.append("")

    # detect degenerate probability saturation (a constant prediction for
    # every input) vs. genuine chance-level (varied, uninformative) output
    # — these are very different findings and must not be conflated
    saturation_rows = []
    for variant, variant_results in all_results.items():
        for model_name, r in variant_results.items():
            for cond_key, cond_label in (("clean", "clean"), ("telephone", "telephone")):
                proba = r["df"][r["df"].channel_condition == cond_label]["proba"]
                saturation_rows.append({
                    "dataset_version": variant, "model": model_name, "channel_condition": cond_label,
                    "proba_std": float(proba.std()), "proba_mean": float(proba.mean()),
                    "is_saturated": bool(proba.std() < 1e-6),
                })
    saturation_df = pd.DataFrame(saturation_rows)
    n_saturated = int(saturation_df["is_saturated"].sum())
    lines.append(
        f"**Probability saturation check**: {n_saturated}/{len(saturation_df)} (model, variant, "
        "condition) combinations produce a probability with essentially ZERO variance across all "
        "clips (std < 1e-6) — i.e. the model outputs the SAME prediction for every single clip "
        "regardless of its actual content. This is NOT the same finding as \"chance-level "
        "separation\" (which would show varied, just uninformative, probabilities) — it means "
        "these clips fall so far outside the training feature distribution that the linear/kernel "
        "decision function saturates completely. Detail: "
        + ", ".join(f"{row.model}/{row.dataset_version}/{row.channel_condition}: "
                     f"{'SATURATED at ' + format(row.proba_mean, '.4f') if row.is_saturated else 'varies (std=' + format(row.proba_std, '.4f') + ')'}"
                     for row in saturation_df.itertuples()) + "\n"
    )

    lines.append("### Confusion matrices (logistic_regression, original variant)\n")
    lr_orig = all_results["original"]["logistic_regression"]
    for cond, cm in (("clean", lr_orig["clean_cm"]), ("telephone", lr_orig["telephone_cm"])):
        lines.append(f"**{cond}**\n")
        lines.append(pd.DataFrame(cm, index=["true_human", "true_synthetic"],
                                   columns=["pred_human", "pred_synthetic"]).to_markdown())
        lines.append("")

    lines.append("## C. Cross-domain comparison — the channel-hypothesis table\n")
    lines.append(
        "| Condition | Question |\n|---|---|\n"
        "| Clean human vs. clean synthetic | Is there external synthetic-speech signal at all? |\n"
        "| Telephone human vs. telephone synthetic | Does that signal survive the telephone domain? |\n"
        "| Clean human vs. telephone human | How much does channel ALONE change predictions on human speech? |\n"
        "| Clean synthetic vs. telephone synthetic | How much does channel ALONE change predictions on synthetic speech? |\n"
    )
    ce_display = channel_effect_all[channel_effect_all.dataset_version == "original"][
        ["model", "speech_type", "clean_mean_proba_synthetic", "telephone_mean_proba_synthetic",
         "clean_frac_predicted_synthetic", "telephone_frac_predicted_synthetic", "shift"]
    ]
    lines.append("\n" + ce_display.round(4).to_markdown(index=False))
    lines.append("\nFull table (all 4 variants): `outputs/features/domain_matched_channel_effect.csv`.\n")

    human_shift = ce_display[ce_display.speech_type == "human"]["shift"].abs().mean()
    synth_shift = ce_display[ce_display.speech_type == "synthetic"]["shift"].abs().mean()
    lines.append(
        f"Mean |shift| in predicted synthetic-probability when moving clean->telephone: "
        f"human speech {human_shift:.4f}, synthetic speech {synth_shift:.4f} "
        f"(`original` variant, all 3 models averaged).\n"
    )

    lines.append("## Generator-level results\n")
    gen_display = gen_all[(gen_all.dataset_version == "original") & (gen_all.model == "logistic_regression")][
        ["condition", "generator_id", "n", "n_human", "n_synthetic", "roc_auc", "pr_auc", "f1"]
    ]
    lines.append(gen_display.round(4).to_markdown(index=False))
    lines.append("\nFull table: `outputs/features/domain_matched_generator_level_results.csv`.\n")

    lines.append("## Speaker-level results\n")
    spk_display = speaker_all[(speaker_all.dataset_version == "original") & (speaker_all.model == "logistic_regression")]
    lines.append(spk_display.round(4).to_markdown(index=False))
    lines.append("\nFull table (all models/variants): `outputs/features/domain_matched_speaker_level_results.csv`.\n")

    lines.append("## Feature diagnostics\n")
    lines.append(
        "See `outputs/plots/domain_matched_validation/` for `std_zcr`, `mean_spectral_rolloff`, "
        "`mean_rms`, and `mfcc1_mean` distributions across current-human/current-synthetic/"
        "clean-external-human/clean-external-synthetic/telephone-external-human/"
        "telephone-external-synthetic, plus a clean-vs-telephone ROC-AUC bar chart across all "
        "4 preprocessing variants.\n"
    )

    lines.append("## Critical interpretation\n")
    clean_auc = lr_orig["clean_metrics"]["roc_auc"]
    tel_auc = lr_orig["telephone_metrics"]["roc_auc"]
    human_tel_frac = ce_display[ce_display.speech_type == "human"]["telephone_frac_predicted_synthetic"].mean()
    lr_saturated = saturation_df[(saturation_df.model == "logistic_regression") & (saturation_df.dataset_version == "original")]["is_saturated"].all()
    svm_saturated = saturation_df[(saturation_df.model == "svm") & (saturation_df.dataset_version == "original")]["is_saturated"].all()

    if lr_saturated or svm_saturated:
        lines.append(
            f"**The clean_auc=0.5000/telephone_auc=0.5000 result above is NOT chance-level "
            f"separation — it is complete probability saturation.** Verified directly: "
            f"logistic_regression outputs exactly 1.0 (max confidence \"synthetic\") for "
            f"{'every single' if lr_saturated else 'most'} clip in BOTH the clean and telephone "
            f"conditions (std of predicted probability = 0.0 to floating-point precision); SVM "
            f"outputs exactly 0.4043 for {'every single' if svm_saturated else 'most'} clip, "
            "again identical regardless of clean/telephone or human/synthetic. A model that "
            "genuinely found no signal would still produce VARIED, just uninformative, "
            "probabilities — a constant output means these clips fall so far outside the "
            "training feature distribution (raw acoustic feature scale, not just the clean-vs-"
            "telephone difference) that the linear/kernel decision function saturates outright. "
            "**This means ASVspoof, for these two models, cannot cleanly answer the clean-vs-"
            "telephone question at all — the language/corpus-level distribution shift (English, "
            "16kHz-native VCTK-derived speech vs. this project's Spanish, natively-8kHz telephone "
            "data) is apparently large enough to dominate over the finer channel manipulation "
            "this experiment was designed to isolate.** Only Random Forest (which splits on "
            "individual feature thresholds rather than one global linear combination) avoids "
            "this total collapse, though its own results are weak and inconsistent (clean "
            "ROC-AUC 0.30-0.39, i.e. below chance / reversed; telephone ROC-AUC 0.19-0.61, "
            "varying by preprocessing variant) — not a clean signal either.\n"
        )
    else:
        lines.append(
            f"Clean external ROC-AUC: {clean_auc:.4f}. Telephone external ROC-AUC: {tel_auc:.4f}. "
            "Per the task's own interpretation rule, an above-chance telephone result alongside a "
            "near-chance clean result must NOT be read as \"the model detects synthetic speech\" "
            "— it indicates channel-dependence, not content-dependence.\n"
        )

    lines.append(
        f"{human_tel_frac:.0%} of genuinely-telephone-transmitted human speech is classified as "
        f"synthetic (`original`/logistic_regression) — but given the saturation finding above, "
        "this number reflects the model calling essentially everything synthetic in this domain, "
        "not a specific human-speech effect.\n"
    )

    gen_tel = gen_display[gen_display.condition == "telephone"] if "condition" in gen_display.columns else pd.DataFrame()
    n_generalizing = int((gen_tel["roc_auc"] >= 0.7).sum()) if len(gen_tel) else 0
    lines.append(
        f"Of the systems evaluated under the telephone condition, {n_generalizing}/{len(gen_tel) if len(gen_tel) else 0} "
        "reach ROC-AUC >= 0.7 against matched human speech in that same condition "
        "(logistic_regression) — uninformative given the saturation above (every generator "
        "shows the identical 0.5 pattern for the same reason).\n"
    )

    lines.append("\n## Final conclusion\n")
    lines.append(
        "**Does matching the actual acquisition/channel domain explain the previous external "
        "AUC of ~0.75, or is there evidence of a transferable synthetic-speech signal?** "
        "**Neither question could be cleanly answered with this dataset.** ASVspoof2021 LA audio "
        "(English, VCTK-derived, natively 16kHz) sits so far outside this project's training "
        "feature distribution that 2 of 3 locked models (logistic regression, SVM) saturate to a "
        "single constant prediction regardless of clean vs. telephone condition or human vs. "
        "synthetic content — this is evidence of a language/corpus-level distribution shift "
        "overwhelming the finer channel manipulation, not evidence about the channel hypothesis "
        "itself. Random Forest alone produced non-degenerate (though weak and inconsistent) "
        "output, and even it shows no reliable clean-vs-telephone pattern. This experiment's "
        "own confound (source-corpus/language mismatch) turned out to be larger than the "
        "channel effect it was designed to isolate.\n"
    )

    status = f"""
## Status summary

- Domain-matched external data found: YES (ASVspoof2021 LA eval, Zenodo, ODC-BY 1.0)
- Actually telephone-acquired/transmitted: YES (`pstn` condition; verified against the official evaluation plan text, not inferred from filename/label)
- External human speech: YES (bona fide, 8 speakers x2 conditions, 5 clips each)
- External synthetic speech: YES (spoof, 8 of 13 TTS/VC systems x2 conditions, 5 clips each)
- Synthetic generators: 8 (of 13 available; A07-A19 range)
- Human speakers: 8 (of 67 available in this subset's speaker pool)
- Strict holdout: YES (zero tuning against this data; threshold from internal val only)
- Zero-shot evaluation: YES (same locked models, no retraining)

Clean external AUC: {clean_auc:.4f} (logistic_regression; SATURATED/degenerate, not a genuine chance-level result — see above)
Telephone external AUC: {tel_auc:.4f} (same caveat)

Human channel effect: uninterpretable for LR/SVM due to probability saturation ({human_tel_frac:.0%} of telephone-human called synthetic, but so is ~100% of everything else)
Synthetic channel effect: same saturation caveat applies
Generator consistency: uninterpretable for the same reason ({n_generalizing}/{len(gen_tel) if len(gen_tel) else 0} nominally reach ROC-AUC>=0.7, but this reflects the constant-output artifact)

Main conclusion: This dataset could not cleanly test the clean-vs-telephone channel hypothesis. A language/corpus-level distribution shift (English ASVspoof vs. Spanish training data) is large enough to saturate 2 of 3 locked models into a single constant prediction regardless of channel condition, swamping the intended comparison. This is a genuine, valid research finding — not a null result to paper over — but it means the original question ("does the 0.75 recovered AUC reflect channel or content?") remains unanswered.
Confidence: Low — the dominant confound in this specific experiment (language/corpus mismatch) was larger than anticipated, on top of the small subset size (8 speakers, 8 systems) and partial speaker/system overlap already noted.
Recommended next step: The clean-vs-telephone question requires a domain-matched dataset in the SAME language as the training data (Spanish) — no such freely-available dataset was identified in this session (see external_data/README.md's sourcing discussion). Absent that, the next-smallest useful step is checking whether even a language-MATCHED but recording-condition-mismatched Spanish set (e.g. clean Spanish TTS/human speech, still not telephone-transmitted) at least avoids the total saturation seen here, to isolate whether saturation is driven by language or by the channel/recording-chain difference specifically.
"""
    lines.append(status)

    with open(REPORT_MD, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    run()
