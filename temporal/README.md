# Temporal / conversational analysis

Turn-level, turn-taking, response-latency, within-turn-pause, and
descriptive human-vs-synthetic comparison for the target speaker
(Channel 0), built on top of the acoustic outputs from `../preprocessing/`.
See `../rules.md` for the project rules and `../preprocessing/README.md`
for the acoustic stage this depends on. No classifier, no ML, no NLP here
— descriptive statistics and plots only.

Uses the 258 recordings with valid JSON turn metadata. The 95 recordings
with empty JSON are excluded from every turn-level output — no turn
boundaries are fabricated for them.

## Running

```bash
source .venv/bin/activate      # same venv as preprocessing/
python -m temporal.pipeline    # turn features -> outputs/features/temporal_*.csv
python -m temporal.compare     # human-vs-synthetic comparison + confounding, appended
                                # to outputs/reports/temporal_analysis_report.md,
                                # plots -> outputs/plots/temporal/
```

Reads (never writes to) `files/`, `outputs/features/recording_level.csv`,
and `outputs/features/timeseries.csv` from the acoustic stage. Does not
modify or overwrite any existing acoustic output.

## Why "target speaker inactive" isn't automatically "a pause"

Channel 0 is one party of a two-party call. Channel 0 being quiet is, in
order of likelihood, one of:

1. **Target-speaker inactivity** — the other speaker (Channel 1) has the
   floor. This is not analyzed as target behavior at all here.
2. **Between-turn gap** — the JSON turn boundaries already delimit this;
   see `inter_turn_gap` / `response_latency` below.
3. **Within-turn silence** — a genuinely low-energy stretch *inside* a
   target turn's own `[start, end)` boundaries. Only this case is
   reported as a candidate pause (Step 7 below), and even then only as
   an *acoustic* observation, never labeled "hesitation."

## Modules

- `turns_io.py` — `load_and_validate_turns(anon_id, duration_s)` parses
  one JSON file, drops (but reports) individually invalid turns, sorts
  turns chronologically, and flags same-channel overlaps. `validate_all`
  runs this over every recording for the validation report.
- `turn_features.py` — all the feature math: target/other turn duration
  stats, speaking ratio, turn-taking transition counts, response
  latency, within-turn candidate-pause detection, and temporal
  variability (CV, quantiles, short/long turn counts). All thresholds
  are named module-level constants with a comment explaining the choice.
- `pipeline.py` — orchestrates the above across the 258 valid-turn
  recordings and writes the outputs listed below.
- `compare.py` — descriptive (non-ML) human-vs-synthetic comparison and
  confounding checks, appended to the same analysis report, plus plots.

## Key methodology notes

- **`target_speaking_ratio`** divides by the *conversation span*
  (earliest turn start to latest turn end, across both channels), not
  the raw wav `duration_s` — a long silent lead-in/trailer would
  otherwise understate it. `target_speaking_ratio_full_duration` (wav
  duration denominator) is kept alongside for reference/confound checks.
- **`inter_turn_gap`** = `next_turn.start - previous_turn.end` for every
  consecutive pair in the chronologically-merged (both channels) turn
  sequence. Negative values (overlapping speech) are kept, never
  clamped to zero.
- **`response_latency`** is the same gap, restricted to
  other→target transitions specifically (the target taking the floor
  right after the other speaker). Also negative when the target starts
  before the other speaker finishes (interruption/overlap) — preserved,
  and separately counted (`response_latency_n_overlapping`,
  `..._prop_overlapping`). "Near zero" uses a ±0.05s tolerance
  (`NEAR_ZERO_GAP_TOLERANCE_S`).
- **Within-turn candidate pauses**: a frame (32ms, matching the acoustic
  pipeline's frame hop) counts as low-energy if its RMS is below the
  20th percentile (`PAUSE_RMS_PERCENTILE`) of RMS values seen across
  *this recording's own target turns* — i.e. relative to how loud this
  speaker is when they are the one talking, not the whole recording
  (which is mostly silence from Channel 0's perspective whenever the
  other speaker has the floor). A contiguous low-energy run must last
  ≥150ms (`MIN_PAUSE_DURATION_S`) to count, and is clipped to stay
  strictly inside its turn. Computed on the **original**, not denoised,
  Channel-0 signal. These are candidate *acoustic* pauses only.
- **Short/long target turns**: fixed thresholds, <0.5s / >10.0s
  (`SHORT_TURN_THRESHOLD_S` / `LONG_TURN_THRESHOLD_S`) — simple and
  fixed rather than data-driven, per rules.md's "don't overengineer".
- **Coefficient of variation (CV)** = `std / |mean|`. For a near-zero-mean
  quantity (response latency can average close to 0) this is numerically
  unstable — treat a large CV on a small-magnitude mean with caution.

## Outputs (`outputs/`, git-ignored, regenerate any time)

### `outputs/features/temporal_recording_level.csv` — one row per recording (258 rows)

`anon_id, label, split, duration_s, has_turn_data` (always `True` here),
then: target turn stats (`target_turn_count`, `target_total_speaking_time`,
`target_mean/median/std/min/max_turn_duration`), other-speaker stats
(same shape, `other_*`), `conversation_span_s`, `target_speaking_ratio`,
`target_speaking_ratio_full_duration`, transition counts
(`total_transitions`, `target_to_other_transitions`,
`other_to_target_transitions`, `target_to_target_transitions`,
`other_to_other_transitions`, `proportion_target_followed_by_other`,
`proportion_other_followed_by_target`), `inter_turn_gap_*` summary,
`response_latency_*` summary (incl. `_n`, `_n_positive`, `_n_near_zero`,
`_n_overlapping` and their proportions), `within_turn_pause_*` summary,
and temporal-variability columns (`*_cv`, `*_q10..q90`,
`n_short_target_turns`, `n_long_target_turns`).

### `outputs/features/temporal_events.csv` — one row per turn (either channel)

`anon_id, label, event_type ("turn"), speaker ("target"/"other"), start,
end, duration, previous_speaker, next_speaker, gap_before, gap_after`.
A target turn with `previous_speaker == "other"` has `gap_before` equal
to its response latency — a separate "transition"/"response" event type
would just duplicate this table under another name, so it wasn't added.

### `outputs/features/temporal_within_turn_pauses.csv` — one row per candidate pause

`anon_id, label, turn_start, turn_end, pause_start, pause_end,
pause_duration`. Kept as its own table (rather than squeezed into the
turn-shaped event table above) since a pause is a sub-interval of a turn,
not a turn itself.

### `outputs/reports/temporal_processing_report.csv` and `temporal_analysis_report.md`

CSV: per-recording ok/failed status. Markdown: JSON validation counts,
settings used, failures, the Step 12 confounding checks, and the Step 11
human-vs-synthetic descriptive comparison (written by `pipeline.py`,
appended to by `compare.py`).

### `outputs/plots/temporal/*.png`

Boxplots (recording-level) and pooled histograms/ECDFs (per-turn /
per-transition / per-pause) comparing human vs. synthetic, plus one
confound-check boxplot (`box_duration_confound.png`).
