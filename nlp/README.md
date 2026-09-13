# NLP / linguistic analysis

> **Status: transcription is currently partial (122/353 recordings) and
> intentionally paused** — see `../outputs/transcripts/STATUS.md`. ASR
> is a secondary/exploratory layer for this project; the first baseline
> model (`../modeling/`) does not depend on it. `nlp_pipeline.py`/
> `compare.py` below still work on whatever subset is transcribed.

Local transcription (Channel 0 and Channel 1, separately) plus simple,
rule-based linguistic features and descriptive human/synthetic/AI-agent
comparisons. See `../rules.md` for project rules, `../preprocessing/README.md`
for the acoustic stage, and `../temporal/README.md` for the turn-taking
stage this one reuses. No embeddings, no LLM-based features, no
classifier — everything here is a plain count/ratio computed from
transcript text plus the existing temporal analysis.

## Dataset fact this stage treats as ground truth

Per the task instructions (not inferred from audio): **Channel 1 is
always a confirmed AI agent** — a separate, dataset-specified group,
not "context." The three groups analyzed throughout are:

1. **Channel 0, `human`** (label from `manifest.csv`)
2. **Channel 0, `synthetic`** (label from `manifest.csv`)
3. **Channel 1, `confirmed_ai_agent`** (dataset specification, not manifest.csv)

`speaker_type` is the column carrying this 3-way distinction everywhere
in this stage. `label`/`label_source` are also kept explicit, per the
task's schema, so the provenance of each grouping is traceable.

## Setup

Same `.venv` as the other stages, plus one new dependency:

```bash
source .venv/bin/activate
pip install -r requirements.txt   # adds faster-whisper
```

`faster-whisper` runs the "small" Whisper model **locally on CPU**
(int8 quantized) — no audio is sent to any external/cloud transcription
API. Model weights are downloaded once (a public HuggingFace repo
download, not per-recording traffic) and cached under `~/.cache`.

## Running

```bash
python -m nlp.transcribe_pipeline   # slow (hours) — see below
python -m nlp.nlp_pipeline
python -m nlp.compare
```

### Why "small", not a faster/smaller model

A quick side-by-side on real turns showed the smaller "base" model
introducing clear transcription errors ("recordo" instead of "recuerdo",
garbled clauses) that would directly contaminate every lexical feature
downstream. "small" costs roughly 3x the compute but produces materially
cleaner Spanish transcripts. Given this stage's output feeds a
linguistic-differences analysis, transcript quality was prioritized over
speed. Transcribing all 353 recordings x 2 channels this way takes on
the order of hours on CPU — `transcribe_pipeline.py` writes
`recordings.csv`/`turns.csv` incrementally (flushed after every
recording) specifically so an interruption doesn't lose already-completed
work; re-running resumes from scratch (it overwrites), but partial
output up to the interruption point remains on disk if needed.

## Modules

- `asr.py` — faster-whisper wrapper. Model loaded once per worker
  process (loading takes ~20-40s, reused for every subsequent call).
  Audio is resampled from the dataset's native 8000 Hz to the 16000 Hz
  Whisper expects (`scipy.signal.resample_poly`). Language is forced to
  Spanish. Returns text plus ASR provenance (`avg_logprob`,
  `no_speech_prob`, detected language + probability).
- `transcribe_pipeline.py` — orchestrates transcription across all 353
  recordings. Reuses `temporal.turns_io.load_and_validate_turns` (not
  reimplemented) to get the same validated turn boundaries the temporal
  stage uses. For the 258 valid-turn recordings, transcribes each turn
  as its own segment; for the 95 empty-JSON recordings, transcribes each
  whole channel once (no turn boundaries fabricated).
- `text_features.py` — plain regex tokenization (Spanish, letters only)
  and hand-built Spanish word lists for fillers/articles/pronouns/
  conjunctions/prepositions. Explicitly a lexicon-based approximation,
  not a trained POS tagger — see "Known limitations" below.
- `nlp_pipeline.py` — applies `text_features` to every transcript and
  joins turn-level features with the existing `temporal_events.csv`
  (for `gap_before`/`previous_speaker`/etc. — not recomputed here).
- `compare.py` — three-way descriptive comparison (human C0 / synthetic
  C0 / confirmed-AI C1), a conversation-level paired analysis of whether
  the AI agent's language changes by partner type, and an ASR-quality
  confound check.

## Known limitations (read before interpreting results)

- **Tokenization/POS is lexicon-based, not statistical.** `este`,
  `pues`, and `que` in particular are ambiguous in Spanish (`este` is
  both a filler and a demonstrative pronoun; `pues` is both a filler and
  a conjunction; `que` is both a conjunction and a relative pronoun) —
  they are counted in every category they plausibly belong to rather
  than disambiguated by context. Category counts should be read as
  "words that commonly serve this function," not a precise parse.
- **Type-token ratio (TTR) is length-sensitive** — shorter turns
  mechanically produce higher TTR. Comparisons across groups with very
  different turn-length distributions (see the temporal-stage findings)
  should account for this rather than reading TTR differences at face
  value.
- **ASR errors are a real confound**, not just a caveat — see
  `compare.py`'s confound section and the report for concrete counts of
  low-confidence/possible-hallucination segments. A linguistic
  difference between groups could partly reflect transcription accuracy
  differences rather than (or in addition to) real speech differences.
- Fillers are counted as **linguistic markers**, never automatically
  labeled "hesitation" or "true hesitation" — that would require
  prosodic/pause evidence this module doesn't use.

## Outputs

### `outputs/transcripts/` (raw ASR output + provenance)

- `recordings.csv` — one row per (`anon_id`, `channel`): `anon_id,
  channel, speaker_type, label, split, label_source, duration_s,
  has_turn_data, transcription_status (turn_level/whole_recording),
  n_turns, transcribed_speaking_time_s, full_text, asr_model,
  mean_avg_logprob, mean_no_speech_prob`.
- `turns.csv` — one row per transcribed turn (258 valid-turn recordings
  only): `anon_id, channel, speaker_type, label, label_source,
  turn_index, start, end, duration, text, asr_model, avg_logprob,
  no_speech_prob, language, language_probability`.

### `outputs/features/` (derived linguistic features)

- `nlp_turns.csv` — per-turn lexical features (word/char/unique-word
  counts, TTR, filler/repetition counts, function-word counts,
  `words_per_second`, consecutive-turn deltas `word_count_delta_prev` /
  `ttr_delta_prev`) joined with `previous_speaker`/`next_speaker`/
  `gap_before`/`gap_after` from the existing `temporal_events.csv`.
- `nlp_recordings.csv` — the same lexical features computed on each
  recording-channel's full transcript, plus `words_per_minute` and
  per-turn aggregates (`mean_words_per_turn`, `mean_ttr_per_turn`, ...).
- `nlp_top_ngrams.csv` — most common unigrams/bigrams/trigrams per
  `speaker_type` group (exploratory look, not a per-recording feature).

### `outputs/reports/`

- `transcription_report.csv`/`.md` — per-recording ok/failed status,
  ASR settings used.
- `nlp_analysis_report.md` — the three-way comparison, the AI-agent
  paired-by-partner-type analysis, and the ASR-quality confound section.

### `outputs/plots/nlp/*.png`

Boxplots/histograms/ECDFs comparing the three `speaker_type` groups, plus
partner-type comparisons for the AI agent's behavior.
