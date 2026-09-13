"""
Descriptive human-vs-synthetic comparison + confounding checks for the
temporal/conversational features.

Purely descriptive: no classifier, no hypothesis testing framed as
"detecting AI", no feature selection. Appends its findings to
outputs/reports/temporal_analysis_report.md (written first by
temporal/pipeline.py) and writes plots to outputs/plots/temporal/.

Run with: python -m temporal.compare   (after python -m temporal.pipeline)
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.audio_io import JSON_DIR, MANIFEST_CSV
from preprocessing.pipeline import REPORTS_DIR
from temporal.pipeline import (
    TEMPORAL_RECORDING_LEVEL_CSV,
    TEMPORAL_EVENTS_CSV,
    TEMPORAL_PAUSES_CSV,
    TEMPORAL_REPORT_MD,
)

PLOTS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "plots" / "temporal"

RECORDING_LEVEL_FEATURES = [
    "target_turn_count", "target_speaking_ratio", "target_mean_turn_duration",
    "target_total_speaking_time", "other_turn_count",
    "response_latency_mean", "response_latency_prop_overlapping",
    "inter_turn_gap_mean", "within_turn_pause_count",
    "within_turn_pause_ratio", "within_turn_pause_mean_duration",
    "target_turn_duration_cv", "response_latency_cv", "inter_turn_gap_cv",
]


def descriptive_table(df: pd.DataFrame, value_col: str, label_col: str = "label") -> pd.DataFrame:
    g = df.groupby(label_col)[value_col]
    out = g.agg(["count", "mean", "median", "std", "min", "max"])
    for q in (0.1, 0.25, 0.75, 0.9):
        out[f"q{int(q*100)}"] = g.quantile(q)
    return out


def boxplot_by_label(df, value_col, title, path, label_col="label"):
    labels = sorted(df[label_col].dropna().unique())
    data = [df.loc[df[label_col] == lab, value_col].dropna().values for lab in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(value_col)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def hist_by_label(df, value_col, title, path, label_col="label", bins=40, xlim=None):
    labels = sorted(df[label_col].dropna().unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    for lab in labels:
        vals = df.loc[df[label_col] == lab, value_col].dropna().values
        if xlim:
            vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
        ax.hist(vals, bins=bins, alpha=0.5, density=True, label=lab)
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel("density")
    if xlim:
        ax.set_xlim(xlim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def ecdf_by_label(df, value_col, title, path, label_col="label", xlim=None):
    labels = sorted(df[label_col].dropna().unique())
    fig, ax = plt.subplots(figsize=(7, 4))
    for lab in labels:
        vals = np.sort(df.loc[df[label_col] == lab, value_col].dropna().values)
        if xlim:
            vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, label=lab)
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel("ECDF")
    if xlim:
        ax.set_xlim(xlim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def confounding_checks(rec: pd.DataFrame) -> str:
    lines = ["## Step 12 — confounding checks\n"]

    lines.append("### Recording duration by label (duration_s)\n")
    lines.append(descriptive_table(rec, "duration_s").round(2).to_markdown())
    lines.append("")

    lines.append("### Target turn count by label\n")
    lines.append(descriptive_table(rec, "target_turn_count").round(2).to_markdown())
    lines.append("")

    lines.append("### train/val split by label (of the 258 recordings with turn data)\n")
    ct = pd.crosstab(rec["label"], rec["split"])
    lines.append(ct.to_markdown())
    lines.append("")

    # has_turn_data proportion by label across ALL 353 recordings, not just the 258 here
    with open(MANIFEST_CSV, newline="") as f:
        manifest_rows = list(csv.DictReader(f))
    full = pd.DataFrame(manifest_rows)
    full["has_turn_data"] = full["anon_id"].apply(
        lambda a: (JSON_DIR / f"{a}.json").exists() and (JSON_DIR / f"{a}.json").stat().st_size > 0
    )
    lines.append("### has_turn_data proportion by label (all 353 recordings)\n")
    prop = full.groupby("label")["has_turn_data"].agg(["sum", "count"])
    prop["proportion_with_turn_data"] = (prop["sum"] / prop["count"]).round(3)
    lines.append(prop.to_markdown())
    lines.append("")

    # simple outlier flags (IQR rule) on target_turn_count and duration_s, per label
    lines.append("### Outlier counts (IQR rule: outside Q1-1.5*IQR .. Q3+1.5*IQR), per label\n")
    for col in ("duration_s", "target_turn_count", "target_speaking_ratio"):
        rows = []
        for lab, g in rec.groupby("label"):
            q1, q3 = g[col].quantile(0.25), g[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((g[col] < lo) | (g[col] > hi)).sum())
            rows.append({"label": lab, "column": col, "n_outliers": n_out, "n_total": len(g)})
        lines.append(pd.DataFrame(rows).to_markdown(index=False))
        lines.append("")

    lines.append(
        "These are observations only — no automatic exclusion or correction was applied to the "
        "dataset or the features above; any of these factors (duration, turn count, split "
        "imbalance, missing-turn-data rate) should be kept in mind when reading the label "
        "comparison below, since a difference in a temporal feature could partly reflect one of "
        "these structural factors rather than (or in addition to) human-vs-synthetic speech "
        "behavior itself.\n"
    )
    return "\n".join(lines)


def comparison_section(rec: pd.DataFrame, events: pd.DataFrame, pauses: pd.DataFrame) -> str:
    lines = ["## Step 11 — human vs. synthetic descriptive comparison\n"]
    lines.append(
        "Descriptive only: counts/mean/median/std/quantiles and plots. No classifier was "
        "trained, no accuracy was computed, and no feature is claimed to 'detect' synthetic "
        "speech — differences below are reported as observed distributional differences that "
        "may warrant further investigation.\n"
    )

    lines.append("### Recording-level features\n")
    for col in RECORDING_LEVEL_FEATURES:
        lines.append(f"**{col}**\n")
        lines.append(descriptive_table(rec, col).round(4).to_markdown())
        lines.append("")

    target_turns = events[(events.speaker == "target") & (events.event_type == "turn")]
    lines.append("### Pooled per-turn target turn duration (all target turns, all recordings)\n")
    lines.append(descriptive_table(target_turns, "duration").round(4).to_markdown())
    lines.append("")

    response_events = target_turns[target_turns.previous_speaker == "other"]
    lines.append("### Pooled per-transition response latency (other -> target only)\n")
    lines.append(descriptive_table(response_events, "gap_before").round(4).to_markdown())
    lines.append("")

    all_gaps = events.dropna(subset=["gap_before"])
    lines.append("### Pooled inter-turn gaps (all consecutive-turn pairs, both channels)\n")
    lines.append(descriptive_table(all_gaps, "gap_before").round(4).to_markdown())
    lines.append("")

    if len(pauses):
        lines.append("### Pooled within-turn candidate-pause duration\n")
        lines.append(descriptive_table(pauses, "pause_duration").round(4).to_markdown())
        lines.append("")

    # aggregate overlap rate (sum-based, more stable than averaging per-recording proportions)
    lines.append("### Aggregate response-latency overlap rate by label\n")
    agg = rec.groupby("label")[["response_latency_n_overlapping", "response_latency_n"]].sum()
    agg["overlap_rate"] = (agg["response_latency_n_overlapping"] / agg["response_latency_n"]).round(4)
    lines.append(agg.to_markdown())
    lines.append("")

    return "\n".join(lines)


def make_plots(rec: pd.DataFrame, events: pd.DataFrame, pauses: pd.DataFrame):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    boxplot_by_label(rec, "target_turn_count", "Target turn count by label", PLOTS_DIR / "box_target_turn_count.png")
    boxplot_by_label(rec, "target_speaking_ratio", "Target speaking ratio by label", PLOTS_DIR / "box_target_speaking_ratio.png")
    boxplot_by_label(rec, "target_mean_turn_duration", "Mean target turn duration by label", PLOTS_DIR / "box_target_mean_turn_duration.png")
    boxplot_by_label(rec, "response_latency_mean", "Mean response latency by label", PLOTS_DIR / "box_response_latency_mean.png")
    boxplot_by_label(rec, "inter_turn_gap_mean", "Mean inter-turn gap by label", PLOTS_DIR / "box_inter_turn_gap_mean.png")
    boxplot_by_label(rec, "within_turn_pause_ratio", "Within-turn pause ratio by label", PLOTS_DIR / "box_within_turn_pause_ratio.png")
    boxplot_by_label(rec, "target_turn_duration_cv", "Target turn duration CV by label", PLOTS_DIR / "box_target_turn_duration_cv.png")

    # confound checks
    boxplot_by_label(rec, "duration_s", "Recording duration by label (confound check)", PLOTS_DIR / "box_duration_confound.png")

    target_turns = events[(events.speaker == "target") & (events.event_type == "turn")]
    hist_by_label(target_turns, "duration", "Target turn duration (pooled)", PLOTS_DIR / "hist_target_turn_duration.png", xlim=(0, 20))
    ecdf_by_label(target_turns, "duration", "Target turn duration ECDF (pooled)", PLOTS_DIR / "ecdf_target_turn_duration.png", xlim=(0, 20))

    response_events = target_turns[target_turns.previous_speaker == "other"]
    hist_by_label(response_events, "gap_before", "Response latency (pooled, other->target)", PLOTS_DIR / "hist_response_latency.png", xlim=(-5, 10))
    ecdf_by_label(response_events, "gap_before", "Response latency ECDF (pooled)", PLOTS_DIR / "ecdf_response_latency.png", xlim=(-5, 10))

    if len(pauses):
        hist_by_label(pauses, "pause_duration", "Within-turn candidate pause duration (pooled)", PLOTS_DIR / "hist_within_turn_pause_duration.png", xlim=(0, 3))


def run():
    rec = pd.read_csv(TEMPORAL_RECORDING_LEVEL_CSV)
    events = pd.read_csv(TEMPORAL_EVENTS_CSV)
    pauses = pd.read_csv(TEMPORAL_PAUSES_CSV)

    print(f"recording-level rows: {len(rec)}, events: {len(events)}, pauses: {len(pauses)}")

    confound_md = confounding_checks(rec)
    compare_md = comparison_section(rec, events, pauses)

    make_plots(rec, events, pauses)
    plot_list = "\n".join(f"- outputs/plots/temporal/{p.name}" for p in sorted(PLOTS_DIR.glob("*.png")))

    with open(TEMPORAL_REPORT_MD, "a") as f:
        f.write("\n\n" + confound_md + "\n\n" + compare_md)
        f.write(f"\n## Plots generated\n\n{plot_list}\n")

    print(f"Appended comparison + confounding sections to {TEMPORAL_REPORT_MD}")
    print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    run()
