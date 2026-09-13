"""
Evaluates the baseline models (modeling/baseline.py): accuracy,
precision, recall, F1, ROC-AUC, confusion matrix on train AND val
(clearly labeled — val is the headline number, train is shown only to
gauge over/under-fitting). Also a permutation-importance-based feature
family summary (acoustic / MFCC / temporal) — descriptive, not causal.

Run with: python -m modeling.evaluate   (after modeling.prepare_dataset)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay,
)
from sklearn.inspection import permutation_importance

from modeling.baseline import run as train_models
from modeling.prepare_dataset import FEATURES_DIR, REPORTS_DIR, FEATURE_FAMILIES_CSV

PLOTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "plots" / "modeling"
EVAL_REPORT_MD = REPORTS_DIR / "modeling_baseline_report.md"
FEATURE_IMPORTANCE_CSV = FEATURES_DIR / "modeling_feature_importance.csv"
UNIVARIATE_SEPARABILITY_CSV = FEATURES_DIR / "modeling_univariate_separability.csv"


def univariate_separability_section(data) -> tuple:
    """
    For each feature alone, how well does it separate human vs synthetic
    on the TRAIN split (val never touched here)? |AUC - 0.5| + 0.5, so a
    feature that separates "backwards" still shows as informative.
    Computed because the baseline models below turn out to score a
    perfect (or near-perfect) ROC-AUC — when that happens, permutation
    importance stops being informative (there's no room left to drop
    from a ceiling score, especially with many correlated features
    compensating for each other), so this simpler, more transparent
    view is what actually explains WHY the models separate the classes
    so easily.
    """
    lines = ["## Univariate feature separability (train split only)\n"]
    families = pd.read_csv(FEATURE_FAMILIES_CSV).set_index("feature")["family"].to_dict()

    X, y = data["X_train"], data["y_train"]
    rows = []
    for c in data["feature_cols"]:
        sub = pd.concat([X[c], y.rename("y")], axis=1).dropna()
        if sub["y"].nunique() < 2 or len(sub) < 5:
            continue
        auc = roc_auc_score(sub["y"], sub[c])
        rows.append({"feature": c, "family": families.get(c, "unknown"), "auc": max(auc, 1 - auc)})

    sep_df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    sep_df.to_csv(UNIVARIATE_SEPARABILITY_CSV, index=False)

    lines.append("### Top 15 single features by univariate AUC (higher = better human/synthetic separation alone)\n")
    lines.append(sep_df.head(15).round(4).to_markdown(index=False))
    lines.append("")

    family_auc = sep_df.groupby("family")["auc"].agg(["mean", "max", "count"]).round(4)
    lines.append("### Univariate AUC by feature family (mean/max across that family's features)\n")
    lines.append(family_auc.to_markdown())
    lines.append("")

    return "\n".join(lines), sep_df

N_PERMUTATION_REPEATS = 10
RANDOM_STATE = 42


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
    }


def metrics_section(fitted, data) -> tuple:
    lines = ["## Baseline model performance\n"]
    lines.append(
        "label: 1 = synthetic, 0 = human (manifest.csv). **val is the headline split** — "
        "train metrics are shown only to check for over/under-fitting, not as a result.\n"
    )

    all_rows = []
    confusions = {}
    for name, model in fitted.items():
        for split_name, X, y in (
            ("train", data["X_train"], data["y_train"]),
            ("val", data["X_val"], data["y_val"]),
        ):
            proba = model.predict_proba(X)[:, 1]
            pred = model.predict(X)
            m = compute_metrics(y, pred, proba)
            m["model"] = name
            m["split"] = split_name
            all_rows.append(m)
            if split_name == "val":
                confusions[name] = confusion_matrix(y, pred)

    metrics_df = pd.DataFrame(all_rows)[["model", "split", "n", "accuracy", "precision", "recall", "f1", "roc_auc"]]
    lines.append(metrics_df.round(4).to_markdown(index=False))
    lines.append("")

    lines.append("### Confusion matrices (val split, rows=true [human,synthetic], cols=predicted [human,synthetic])\n")
    for name, cm in confusions.items():
        lines.append(f"**{name}**\n")
        lines.append(pd.DataFrame(cm, index=["true_human", "true_synthetic"], columns=["pred_human", "pred_synthetic"]).to_markdown())
        lines.append("")

    return "\n".join(lines), metrics_df, confusions


def feature_importance_section(fitted, data) -> tuple:
    lines = ["## Feature importance (permutation importance, val split, scoring=roc_auc)\n"]
    lines.append(
        "Same methodology (permutation importance) used for all three models so importances "
        "are comparable across model types. Descriptive only — this says a feature's values "
        "matter to THIS model's val-set predictions, not that it causes or detects anything.\n"
    )

    families = pd.read_csv(FEATURE_FAMILIES_CSV).set_index("feature")["family"].to_dict()

    all_importances = []
    for name, model in fitted.items():
        result = permutation_importance(
            model, data["X_val"], data["y_val"],
            scoring="roc_auc", n_repeats=N_PERMUTATION_REPEATS, random_state=RANDOM_STATE,
        )
        for feat, imp in zip(data["feature_cols"], result.importances_mean):
            all_importances.append({
                "model": name, "feature": feat,
                "family": families.get(feat, "unknown"), "importance": imp,
            })

    imp_df = pd.DataFrame(all_importances)
    imp_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    lines.append("### Top 10 individual features per model\n")
    for name, g in imp_df.groupby("model"):
        top = g.sort_values("importance", ascending=False).head(10)
        lines.append(f"**{name}**\n")
        lines.append(top[["feature", "family", "importance"]].round(5).to_markdown(index=False))
        lines.append("")

    lines.append("### Mean importance by feature family, per model\n")
    family_summary = imp_df.groupby(["model", "family"])["importance"].mean().unstack().round(5)
    lines.append(family_summary.to_markdown())
    lines.append("")

    return "\n".join(lines), imp_df


def make_plots(fitted, data, confusions, sep_df):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for feat in sep_df.head(6)["feature"]:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        vals_h = data["X_train"].loc[data["y_train"] == 0, feat].dropna()
        vals_s = data["X_train"].loc[data["y_train"] == 1, feat].dropna()
        ax.boxplot([vals_h, vals_s], tick_labels=["human", "synthetic"], showmeans=True)
        ax.set_title(f"{feat} by label (train split)")
        ax.set_ylabel(feat)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"box_{feat}_by_label.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in fitted.items():
        proba = model.predict_proba(data["X_val"])[:, 1]
        RocCurveDisplay.from_predictions(data["y_val"], proba, name=name, ax=ax)
    ax.set_title("ROC curves (val split)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curves_val.png", dpi=120)
    plt.close(fig)

    for name, cm in confusions.items():
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1], ["pred_human", "pred_synthetic"])
        ax.set_yticks([0, 1], ["true_human", "true_synthetic"])
        ax.set_title(f"Confusion matrix — {name} (val)")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"confusion_matrix_{name}.png", dpi=120)
        plt.close(fig)

    imp_df = pd.read_csv(FEATURE_IMPORTANCE_CSV)
    family_summary = imp_df.groupby(["model", "family"])["importance"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4))
    family_summary.plot(kind="bar", ax=ax)
    ax.set_ylabel("mean permutation importance (roc_auc drop)")
    ax.set_title("Feature-family importance by model")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "feature_family_importance.png", dpi=120)
    plt.close(fig)


def run():
    fitted, data = train_models()

    metrics_md, metrics_df, confusions = metrics_section(fitted, data)
    ceiling = (metrics_df["roc_auc"] >= 0.999).all()
    caveat = ""
    if ceiling:
        caveat = (
            "\n> **Caveat — read before trusting these numbers.** Every model scores at or near "
            "a perfect ROC-AUC on BOTH train and val, and a plain 5-fold cross-validation "
            "*within the train split alone* (val never touched) is also a perfect 1.0000 even "
            "under heavy L2 regularization. No single feature perfectly separates the classes "
            "(checked directly), so this is not the simple kind of leak where one column encodes "
            "the label. It IS consistent with the human and synthetic recordings in this dataset "
            "occupying near-non-overlapping regions of acoustic/MFCC/temporal feature space — "
            "most plausibly because the two groups were captured/generated through different "
            "recording or post-processing pipelines (e.g. different channel/noise-floor/loudness "
            "characteristics), not because the models found subtle synthetic-speech artifacts a "
            "real detector could rely on in the wild. This is a well-known failure mode in "
            "synthetic-audio-detection research (\"shortcut learning\" on recording-condition "
            "differences rather than the actual target). Treat this baseline as a description of "
            "this dataset, not evidence of real-world/live detection capability, until the "
            "recording/generation pipeline difference is investigated directly.\n"
        )

    sep_md, sep_df = univariate_separability_section(data)
    importance_md, imp_df = feature_importance_section(fitted, data)
    if ceiling:
        importance_md += (
            "\n**Why importances above are all ~0:** permutation importance measures the score "
            "DROP when one feature is shuffled. With ROC-AUC already at the 1.0 ceiling and many "
            "redundant/correlated features able to compensate for each other, shuffling any single "
            "feature barely moves the score. The univariate separability table above is the more "
            "informative view of what's actually driving these results.\n"
        )
    make_plots(fitted, data, confusions, sep_df)

    with open(EVAL_REPORT_MD, "w") as f:
        f.write("# Baseline modeling report (acoustic + MFCC + temporal, Channel 0 only)\n\n")
        f.write(metrics_md + caveat + "\n\n" + sep_md + "\n\n" + importance_md)
        f.write(f"\n## Plots\n\n- outputs/plots/modeling/roc_curves_val.png\n"
                f"- outputs/plots/modeling/confusion_matrix_<model>.png\n"
                f"- outputs/plots/modeling/feature_family_importance.png\n"
                f"- outputs/plots/modeling/box_<feature>_by_label.png (top separating features)\n")

    print(f"Wrote {EVAL_REPORT_MD}")
    print(metrics_df.round(4).to_string(index=False))
    if ceiling:
        print("\nWARNING: near-perfect performance detected — see caveat in the report before trusting these results.")
    return metrics_df


if __name__ == "__main__":
    run()
