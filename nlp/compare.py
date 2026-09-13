"""
Descriptive NLP comparisons: Human C0 vs Synthetic C0 vs confirmed-AI C1,
plus paired conversation-level AI-C1-behavior-by-partner-type analysis,
plus ASR-quality confound checks. Purely descriptive — no classifier, no
hypothesis-testing-as-detection framing.

Run with: python -m nlp.compare   (after nlp.nlp_pipeline)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nlp.nlp_pipeline import NLP_TURNS_CSV, NLP_RECORDINGS_CSV, FEATURES_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "outputs" / "reports"
PLOTS_DIR = REPO_ROOT / "outputs" / "plots" / "nlp"
NLP_REPORT_MD = REPORTS_DIR / "nlp_analysis_report.md"
TEMPORAL_EVENTS_CSV = FEATURES_DIR / "temporal_events.csv"
TEMPORAL_RECORDING_LEVEL_CSV = FEATURES_DIR / "temporal_recording_level.csv"

GROUP_ORDER = ["human", "synthetic", "confirmed_ai_agent"]

RECORDING_FEATURES = [
    "word_count", "unique_word_count", "type_token_ratio", "filler_rate",
    "repeated_word_count", "repeated_bigram_count", "function_word_ratio",
    "words_per_second", "words_per_minute", "mean_words_per_turn",
    "mean_ttr_per_turn", "mean_filler_rate_per_turn",
    "mean_avg_logprob", "mean_no_speech_prob",
]


def descriptive_table(df, value_col, group_col="speaker_type"):
    g = df.groupby(group_col)[value_col]
    out = g.agg(["count", "mean", "median", "std", "min", "max"])
    for q in (0.25, 0.75):
        out[f"q{int(q*100)}"] = g.quantile(q)
    order = [x for x in GROUP_ORDER if x in out.index]
    return out.loc[order]


def boxplot_by_group(df, value_col, title, path, group_col="speaker_type"):
    groups = [g for g in GROUP_ORDER if g in df[group_col].unique()]
    data = [df.loc[df[group_col] == g, value_col].dropna().values for g in groups]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.boxplot(data, tick_labels=groups, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(value_col)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def hist_by_group(df, value_col, title, path, group_col="speaker_type", bins=30, xlim=None):
    groups = [g for g in GROUP_ORDER if g in df[group_col].unique()]
    fig, ax = plt.subplots(figsize=(7, 4))
    for g in groups:
        vals = df.loc[df[group_col] == g, value_col].dropna().values
        if xlim:
            vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
        ax.hist(vals, bins=bins, alpha=0.5, density=True, label=g)
    ax.set_title(title)
    ax.set_xlabel(value_col)
    if xlim:
        ax.set_xlim(xlim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def ecdf_by_group(df, value_col, title, path, group_col="speaker_type", xlim=None):
    groups = [g for g in GROUP_ORDER if g in df[group_col].unique()]
    fig, ax = plt.subplots(figsize=(7, 4))
    for g in groups:
        vals = np.sort(df.loc[df[group_col] == g, value_col].dropna().values)
        if xlim:
            vals = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
        y = np.arange(1, len(vals) + 1) / max(len(vals), 1)
        ax.plot(vals, y, label=g)
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel("ECDF")
    if xlim:
        ax.set_xlim(xlim)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def three_way_section(rec: pd.DataFrame, turns: pd.DataFrame) -> str:
    lines = ["## Three-way descriptive comparison (Human C0 / Synthetic C0 / confirmed-AI C1)\n"]
    lines.append(
        "Descriptive only — counts/mean/median/std/quartiles. No classifier, no accuracy, "
        "no claim that any feature 'detects' synthetic speech.\n"
    )
    lines.append("### Recording-level (one row per anon_id x channel)\n")
    for col in RECORDING_FEATURES:
        lines.append(f"**{col}**\n")
        lines.append(descriptive_table(rec, col).round(4).to_markdown())
        lines.append("")

    lines.append("### Pooled per-turn word count (turn-level recordings only)\n")
    lines.append(descriptive_table(turns, "word_count").round(4).to_markdown())
    lines.append("")
    lines.append("### Pooled per-turn type-token ratio\n")
    lines.append(descriptive_table(turns, "type_token_ratio").round(4).to_markdown())
    lines.append("")
    lines.append("### Pooled per-turn filler rate\n")
    lines.append(descriptive_table(turns, "filler_rate").round(4).to_markdown())
    lines.append("")

    return "\n".join(lines)


def build_ai_paired_table(rec: pd.DataFrame) -> pd.DataFrame:
    """
    One row per recording's Channel-1 (confirmed AI agent), tagged with
    the CO-OCCURRING Channel-0 label (human/synthetic) as `partner_type`
    — i.e. is the AI agent talking to a human-labeled or synthetic-labeled
    target speaker in this conversation? Used to check whether the AI
    agent behaves differently depending on who/what it's talking to.
    """
    c0 = rec[rec.channel == 0][["anon_id", "label"]].rename(columns={"label": "partner_type"})
    c1 = rec[rec.channel == 1].copy()
    paired = c1.merge(c0, on="anon_id", how="inner")

    if TEMPORAL_EVENTS_CSV.exists():
        events = pd.read_csv(TEMPORAL_EVENTS_CSV)
        events = events[events.event_type == "turn"]
        # AI (other) turns that follow the target: gap_before is the AI's
        # OWN response latency; negative = AI started before target finished
        ai_after_target = events[(events.speaker == "other") & (events.previous_speaker == "target")]
        latency = ai_after_target.groupby("anon_id")["gap_before"].agg(
            ai_response_latency_mean="mean",
            ai_response_overlap_rate=lambda s: float((s < -0.05).mean()),
        )
        paired = paired.merge(latency, on="anon_id", how="left")

    if TEMPORAL_RECORDING_LEVEL_CSV.exists():
        temporal = pd.read_csv(TEMPORAL_RECORDING_LEVEL_CSV)[["anon_id", "other_mean_turn_duration", "other_turn_count"]]
        temporal = temporal.rename(columns={
            "other_mean_turn_duration": "ai_mean_turn_duration",
            "other_turn_count": "ai_turn_count",
        })
        paired = paired.merge(temporal, on="anon_id", how="left")

    return paired


def paired_section(paired: pd.DataFrame) -> str:
    lines = ["## Conversation-level paired analysis: does the AI agent (Channel 1) behave "
             "differently depending on its partner?\n"]
    lines.append(
        "One row per conversation's Channel-1 (confirmed AI agent), grouped by the "
        "co-occurring Channel-0 label (`partner_type`: human or synthetic). Exploratory "
        "only — no causal claim.\n"
    )
    cols = [
        "word_count", "words_per_second", "mean_words_per_turn", "type_token_ratio",
        "filler_rate", "ai_mean_turn_duration", "ai_turn_count",
        "ai_response_latency_mean", "ai_response_overlap_rate",
    ]
    for col in cols:
        if col not in paired.columns:
            continue
        lines.append(f"**{col}** (AI agent, by partner_type)\n")
        lines.append(descriptive_table(paired, col, group_col="partner_type").round(4).to_markdown())
        lines.append("")
    lines.append(
        "Note: each conversation contributes exactly one row here (not one row per turn), "
        "so these summaries do not overweight recordings with more turns.\n"
    )
    return "\n".join(lines)


def confound_section(rec: pd.DataFrame) -> str:
    lines = ["## ASR-quality confound check\n"]
    lines.append(
        "`mean_avg_logprob` (closer to 0 = more confident) and `mean_no_speech_prob` "
        "(higher = more likely the segment wasn't actually speech) come directly from "
        "faster-whisper. If these differ systematically between groups, apparent "
        "linguistic differences above could partly reflect transcription-quality "
        "differences rather than (or in addition to) real speech differences.\n"
    )
    for col in ("mean_avg_logprob", "mean_no_speech_prob"):
        lines.append(f"**{col}** by group\n")
        lines.append(descriptive_table(rec, col).round(4).to_markdown())
        lines.append("")

    lines.append("### transcription_status counts by group (turn_level vs whole_recording)\n")
    ct = pd.crosstab(rec["speaker_type"], rec["transcription_status"])
    lines.append(ct.to_markdown())
    lines.append(
        "\nNote: `words_per_second`/`words_per_minute` use different denominators for the "
        "two statuses — turn-level recordings divide by summed turn (speaking) duration, "
        "whole-recording rows divide by the full recording duration (which includes silence "
        "while the other channel is talking), mechanically lowering the rate for that subset. "
        "If the turn_level/whole_recording mix differs across groups (see the table above), "
        "that alone could shift a group's average word rate independent of real speech rate.\n"
    )

    # flag likely-hallucinated / low-confidence recordings for manual review
    low_conf = rec[rec["mean_avg_logprob"] < -1.0][["anon_id", "channel", "speaker_type", "mean_avg_logprob", "mean_no_speech_prob"]]
    high_no_speech = rec[rec["mean_no_speech_prob"] > 0.5][["anon_id", "channel", "speaker_type", "mean_avg_logprob", "mean_no_speech_prob"]]
    lines.append(f"Recordings with mean_avg_logprob < -1.0 (low ASR confidence): {len(low_conf)}\n")
    if len(low_conf):
        lines.append(low_conf.round(3).to_markdown(index=False))
        lines.append("")
    lines.append(f"Recordings with mean_no_speech_prob > 0.5 (possible non-speech/silence contamination): {len(high_no_speech)}\n")
    if len(high_no_speech):
        lines.append(high_no_speech.round(3).to_markdown(index=False))
        lines.append("")

    return "\n".join(lines)


def make_plots(rec: pd.DataFrame, turns: pd.DataFrame, paired: pd.DataFrame):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    boxplot_by_group(rec, "word_count", "Word count by group", PLOTS_DIR / "box_word_count.png")
    boxplot_by_group(rec, "type_token_ratio", "Type-token ratio by group", PLOTS_DIR / "box_ttr.png")
    boxplot_by_group(rec, "filler_rate", "Filler rate by group", PLOTS_DIR / "box_filler_rate.png")
    boxplot_by_group(rec, "words_per_second", "Words per second by group", PLOTS_DIR / "box_words_per_second.png")
    boxplot_by_group(rec, "function_word_ratio", "Function-word ratio by group", PLOTS_DIR / "box_function_word_ratio.png")
    boxplot_by_group(rec, "mean_avg_logprob", "ASR mean avg_logprob by group (quality confound)", PLOTS_DIR / "box_asr_confidence.png")

    hist_by_group(turns, "word_count", "Per-turn word count (pooled)", PLOTS_DIR / "hist_turn_word_count.png", xlim=(0, 60))
    ecdf_by_group(turns, "type_token_ratio", "Per-turn TTR ECDF (pooled)", PLOTS_DIR / "ecdf_turn_ttr.png")

    if len(paired):
        boxplot_by_group(paired, "ai_response_latency_mean", "AI response latency by partner type",
                          PLOTS_DIR / "box_ai_response_latency_by_partner.png", group_col="partner_type")
        boxplot_by_group(paired, "words_per_second", "AI words/sec by partner type",
                          PLOTS_DIR / "box_ai_wps_by_partner.png", group_col="partner_type")


def run():
    rec = pd.read_csv(NLP_RECORDINGS_CSV)
    turns = pd.read_csv(NLP_TURNS_CSV)
    print(f"nlp_recordings rows: {len(rec)}, nlp_turns rows: {len(turns)}")

    paired = build_ai_paired_table(rec)
    print(f"AI-paired rows: {len(paired)}")

    three_way_md = three_way_section(rec, turns)
    paired_md = paired_section(paired)
    confound_md = confound_section(rec)

    make_plots(rec, turns, paired)
    plot_list = "\n".join(f"- outputs/plots/nlp/{p.name}" for p in sorted(PLOTS_DIR.glob("*.png")))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(NLP_REPORT_MD, "w") as f:
        f.write("# NLP exploratory analysis report\n\n")
        f.write(f"Recording-level rows: {len(rec)} (2 per recording: channel 0 + channel 1). "
                f"Turn-level rows: {len(turns)}.\n\n")
        f.write(three_way_md + "\n\n" + paired_md + "\n\n" + confound_md)
        f.write(f"\n## Plots generated\n\n{plot_list}\n")

    print(f"Wrote {NLP_REPORT_MD}")
    print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    run()
