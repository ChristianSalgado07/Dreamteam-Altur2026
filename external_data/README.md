# External validation data

A small, fully-documented external evaluation set for testing whether
the human-vs-synthetic classifier trained on the project's main dataset
(`files/`) generalizes to genuinely external speech. This directory is
entirely separate from `files/` — nothing in the original dataset was
touched.

**Two experiments' worth of external data live here**:
1. `raw/openslr61_human/` + `raw/synthetic_gtts/` + `raw/synthetic_espeak/`
   — the original external-validation set (see "Why this data" below).
2. `raw/asvspoof2021_la/` (+ `raw/asvspoof2021_keys/` metadata) — added
   for the **domain-matched validation** follow-up experiment
   (`modeling/domain_matched_validation.py`,
   `outputs/reports/domain_matched_validation_report.md`), which needed
   speech genuinely transmitted through a real telephone channel (not
   just digitally band-limited). See "ASVspoof2021 LA" section below.

## Why this data, not a downloaded "AI-voice-detection" benchmark

Several pre-packaged audio-deepfake-detection datasets exist (ASVspoof,
WaveFake, "In the Wild", Kaggle's "Fake-or-Real", ...). They were
considered but not used here because:

- Several require a Kaggle/registration login this environment cannot
  authenticate through, or ship as tens-of-GB archives not practical to
  fully verify and integrate in this session.
- Most are English-language. The current project's dataset is **Spanish**
  telephone audio — evaluating on a different language would introduce
  a second, uncontrolled confound (phoneme inventory, prosody) on top of
  the one actually being tested (generator identity).
- The task explicitly warns: *"Do not use a dataset merely because it is
  labeled 'AI-generated audio.' Verify what the labels actually mean."*
  For a downloaded benchmark, verifying exactly which TTS/vocoder
  produced each clip requires trusting the benchmark's own documentation
  without independent means to confirm it in this session.

Instead, this external set was **assembled from two sources whose
provenance is fully known and independently verifiable**, matching the
project's language:

### 1. Human speech — OpenSLR SLR61

- **Name**: "Crowdsourced high-quality Argentinian Spanish speech data set"
- **Source**: https://www.openslr.org/61/ (mirror used:
  `https://openslr.trmal.net/resources/61/es_ar_male.zip`)
- **License**: CC BY-SA 4.0 (see `raw/openslr61_human/LICENSE`, copied
  verbatim from the download)
- **Access date**: 2026-09-12 (this session)
- **Content used**: 40 clips, 5 each from 8 distinct male speaker IDs
  (out of 13 available in the male subset), selected deterministically
  (first 5 utterances per speaker in zip order) — NOT the full 552 MB /
  1,820-file archive, to keep this external set small and reviewable.
  The full zip was deleted after extracting the subset (not needed
  again; re-downloadable from the URL above if more is needed later).
- **Speaker IDs**: embedded in the filename (`arm_<speaker_id>_<utt_id>.wav`)
  — used as-is, not inferred.
- **Format**: 48000 Hz, mono, 16-bit PCM, ~2-6s per utterance (isolated
  read sentences, not multi-turn conversation — see Limitations in the
  validation report).
- **Label**: `human` (these are recordings of real speakers reading
  prompted sentences; not synthetic in any sense).

### 2. Synthetic speech — two open-source/free TTS systems, self-generated

Generated directly in this session (`external_data/generate_synthetic.py`)
from 24 original, generic, conversational Spanish sentences (written for
this experiment — no copyrighted text, no PII, not derived from any real
person). Using two systems chosen to be architecturally different from
each other AND from whatever produced this project's own Channel-0
synthetic recordings (a fluent, natural-prosody neural TTS/voice-cloning
system, per the existing transcripts):

- **`gtts_google`**: the `gTTS` Python package, which calls Google
  Translate's public TTS endpoint. A real, modern, widely-used web TTS
  service — architecture/vendor unrelated to this project's synthetic
  source. Output: MP3, 24000 Hz, mono.
- **`espeak_ng`**: `espeak-ng` (installed via Homebrew), a classic
  rule-based formant synthesizer — about as different in kind from a
  neural TTS system as a "generator B" can be, deliberately chosen for
  contrast in the generator-level analysis. Output: WAV, 22050 Hz, mono,
  16-bit PCM.
- 24 clips per generator = 48 synthetic clips total.
- **License**: gTTS output — Google Translate TTS terms (personal/
  research use); espeak-ng is GPL-licensed software, its *output audio*
  carries no additional license restriction. Text prompts are original
  to this project.
- **Label**: `synthetic`, `generator_id` = `gtts_google` or `espeak_ng`
  (never inferred — recorded directly from which script generated it).

### 3. ASVspoof2021 LA evaluation set — for the domain-matched experiment

- **Name**: ASVspoof2021 Challenge — Logical Access (LA) evaluation data
- **Source**: https://zenodo.org/record/4837263 (audio, 7.8GB
  `ASVspoof2021_LA_eval.tar.gz`, MD5 `2abee34d8b0b91159555fc4f016e4562`
  — verified after download) + keys/metadata from
  https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz
- **License**: Open Data Commons Attribution License v1.0 (ODC-BY),
  openly downloadable, no login required
- **Access date**: 2026-09-12 (this session)
- **Recording conditions**: per the official evaluation plan
  (`asvspoof2021_evaluation_plan.pdf`, Section 3, saved during this
  session): bona fide (human) and spoofed (TTS/VC) speech "transmitted
  across either a public switched telephone network (PSTN) or a voice
  over IP (VoIP) network using some particular codec." We use only the
  `none` (untransmitted) and `pstn` (**genuinely PSTN-transmitted** —
  confirmed by the evaluation plan text, not inferred from a filename)
  conditions of the 7 available (`none, alaw, ulaw, g722, gsm, opus,
  pstn`) — the other 5 are simulated codec compression, not confirmed
  real telephone-network transmission.
- **Synthetic generator identity**: 13 distinct TTS/VC systems (A07-A19,
  from the ASVspoof2019 LA attack set), never inferred — taken directly
  from the official `trial_metadata.txt` protocol/keys file.
- **Speakers**: 67 bona fide speaker IDs in the full `none`/`pstn`
  conditions; this experiment uses a small subset (8 speakers, 8
  systems, 5 utterances each, per condition — see
  `select_and_extract_asvspoof.py`, seed=42), not the full 25,938-
  utterance-per-condition set.
- **Format**: FLAC, 16000 Hz, mono, 16-bit (per the challenge spec, "all
  audio data is sampled at a rate of 16 kHz," regardless of the
  underlying transmission's true bandwidth).
- **Language**: English (VCTK-derived) — the current project's dataset
  is Spanish. This is a real, documented confound this experiment could
  not remove; see `outputs/reports/domain_matched_validation_report.md`
  for how it turned out to dominate the results.
- **`none`/`pstn` are NOT the same underlying utterances** (verified:
  zero utterance-ID overlap) — a speaker/system-matched, not
  utterance-matched, comparison; the specific 8-per-condition subsample
  only partially overlaps in speaker/system identity between conditions
  (documented exactly, with IDs, in the validation report).
- Not registered with the ASVspoof consortium — downloaded directly from
  the open Zenodo record for internal research measurement, not
  challenge participation (no leaderboard submission, no scores
  submitted, no comparison to other teams' results).

## What this external set is NOT

- Not multi-turn conversation (no Channel-1 equivalent, no turn-taking) —
  only isolated utterances. Temporal/turn-taking features are therefore
  not applicable; see the validation report for how this was handled.
- Not a large-scale benchmark — 40 human + 48 synthetic (88 clips) is a
  small, quick, fully-inspectable external check, not a substitute for a
  large-scale generalization study.
- Not verified free of *any* confound (recording quality, microphone,
  isolated-sentence vs. phone-call context all differ from the main
  dataset) — see Limitations in `outputs/reports/external_validation_report.md`.

## Directory structure

```
external_data/
  raw/
    openslr61_human/        40 original WAVs (48kHz) + LICENSE + line_index.tsv
    synthetic_gtts/         24 original MP3s (24kHz), gtts_google
    synthetic_espeak/       24 original WAVs (22050Hz), espeak_ng
  processed/                 8kHz mono WAVs, 4 subfolders (original/
                              normalized/telephony/telephony_normalized)
                              — see modeling/external_validation.py
  manifests/
    synthetic_generation_manifest.csv   text + generator per synthetic clip
    external_manifest.csv               the unified manifest (see below)
  generate_synthetic.py       generation script (rerunnable, but gTTS
                               calls a live network endpoint — output
                               will not be byte-identical on a rerun)
  README.md                   this file
```

## Unified manifest

`external_data/manifests/external_manifest.csv` (copied to
`outputs/reports/external_manifest.csv`) — one row per external clip:
`external_id, source_dataset, source_split, label, speaker_id, voice_id,
generator_id, generator_type, duration_s, sample_rate, channels,
original_path, processed_path`. Built by `external_data/build_manifest.py`.
