"""
Computes interpretable NLP features from the transcripts (nlp/transcribe_pipeline.py
output) and joins turn-level features with the existing temporal analysis
(temporal/pipeline.py output) rather than recomputing turn timing/gaps.

Writes:
  outputs/features/nlp_turns.csv       — one row per transcribed turn
  outputs/features/nlp_recordings.csv  — one row per (anon_id, channel)
  outputs/features/nlp_top_ngrams.csv  — most common bigrams/trigrams per speaker_type group

Run with: python -m nlp.nlp_pipeline   (after nlp.transcribe_pipeline)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from nlp.text_features import lexical_features, top_ngrams
from nlp.transcribe_pipeline import RECORDINGS_CSV, TURNS_CSV

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = REPO_ROOT / "outputs" / "features"
TEMPORAL_EVENTS_CSV = FEATURES_DIR / "temporal_events.csv"

NLP_TURNS_CSV = FEATURES_DIR / "nlp_turns.csv"
NLP_RECORDINGS_CSV = FEATURES_DIR / "nlp_recordings.csv"
NLP_TOP_NGRAMS_CSV = FEATURES_DIR / "nlp_top_ngrams.csv"


def build_nlp_turns() -> pd.DataFrame:
    turns = pd.read_csv(TURNS_CSV)
    feats = turns["text"].fillna("").apply(lexical_features).apply(pd.Series)
    out = pd.concat([turns, feats], axis=1)

    out["words_per_second"] = out["word_count"] / out["duration"].replace(0, np.nan)

    # linguistic change across consecutive SAME-CHANNEL turns (turn_index is
    # already chronological within a channel — see temporal/turns_io.py)
    out = out.sort_values(["anon_id", "channel", "turn_index"])
    grp = out.groupby(["anon_id", "channel"])
    out["word_count_delta_prev"] = grp["word_count"].diff()
    out["ttr_delta_prev"] = grp["type_token_ratio"].diff()

    # bring in this turn's gap_before/gap_after/previous_speaker/next_speaker
    # from the existing temporal analysis instead of recomputing it here
    if TEMPORAL_EVENTS_CSV.exists():
        events = pd.read_csv(TEMPORAL_EVENTS_CSV)
        events = events[events.event_type == "turn"].copy()
        events["channel"] = events["speaker"].map({"target": 0, "other": 1})
        events = events.sort_values(["anon_id", "channel", "start"])
        events["turn_index"] = events.groupby(["anon_id", "channel"]).cumcount()
        merge_cols = ["anon_id", "channel", "turn_index", "previous_speaker", "next_speaker", "gap_before", "gap_after"]
        out = out.merge(events[merge_cols], on=["anon_id", "channel", "turn_index"], how="left")

    return out


def build_nlp_recordings(nlp_turns: pd.DataFrame) -> pd.DataFrame:
    recordings = pd.read_csv(RECORDINGS_CSV)
    feats = recordings["full_text"].fillna("").apply(lexical_features).apply(pd.Series)
    out = pd.concat([recordings, feats], axis=1)

    out["words_per_second"] = out["word_count"] / out["transcribed_speaking_time_s"].replace(0, np.nan)
    out["words_per_minute"] = out["words_per_second"] * 60

    # per-turn aggregates, only meaningful for turn_level recordings
    turn_level = nlp_turns.groupby(["anon_id", "channel"]).agg(
        mean_words_per_turn=("word_count", "mean"),
        mean_ttr_per_turn=("type_token_ratio", "mean"),
        mean_repeated_words_per_turn=("repeated_word_count", "mean"),
        mean_filler_rate_per_turn=("filler_rate", "mean"),
    ).reset_index()
    out = out.merge(turn_level, on=["anon_id", "channel"], how="left")

    return out


def build_top_ngrams(nlp_recordings: pd.DataFrame, n_values=(1, 2, 3), top_k=20) -> pd.DataFrame:
    rows = []
    for speaker_type, g in nlp_recordings.groupby("speaker_type"):
        texts = g["full_text"].fillna("").tolist()
        for n in n_values:
            for gram, count in top_ngrams(texts, n=n, top_k=top_k):
                rows.append({
                    "speaker_type": speaker_type,
                    "n": n,
                    "ngram": " ".join(gram),
                    "count": count,
                })
    return pd.DataFrame(rows)


def run():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Computing per-turn NLP features...")
    nlp_turns = build_nlp_turns()
    nlp_turns.to_csv(NLP_TURNS_CSV, index=False)
    print(f"  {len(nlp_turns)} rows -> {NLP_TURNS_CSV}")

    print("Computing per-recording NLP features...")
    nlp_recordings = build_nlp_recordings(nlp_turns)
    nlp_recordings.to_csv(NLP_RECORDINGS_CSV, index=False)
    print(f"  {len(nlp_recordings)} rows -> {NLP_RECORDINGS_CSV}")

    print("Computing top n-grams per speaker_type group...")
    top_ngrams_df = build_top_ngrams(nlp_recordings)
    top_ngrams_df.to_csv(NLP_TOP_NGRAMS_CSV, index=False)
    print(f"  {len(top_ngrams_df)} rows -> {NLP_TOP_NGRAMS_CSV}")

    return {"nlp_turns": len(nlp_turns), "nlp_recordings": len(nlp_recordings)}


if __name__ == "__main__":
    run()
