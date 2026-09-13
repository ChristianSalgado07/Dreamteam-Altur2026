# Acoustic preprocessing (Channel 0)

Foundational, non-ML preprocessing and feature extraction for the target
speaker (Channel 0) of every recording. See `../rules.md` for the project
rules this follows. This stage does **not** do NLP, speech-to-text,
turn-level (temporal/conversational) analysis, or any classification —
just: load Channel 0 → basic signal checks → optional denoising →
acoustic feature extraction.

## Setup

The system Python here is 3.14, which some audio-processing dependencies
(numba, used by librosa) don't yet support. A Python 3.11 virtualenv is
used instead:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`matplotlib` was added to `requirements.txt` — it wasn't there before and
is needed for the exploratory plots (deliverable 6). No other new
dependencies were introduced; the noise-reduction step uses `scipy.signal`
rather than adding a library like `noisereduce`.

## Running

```bash
source .venv/bin/activate
python -m preprocessing.pipeline     # full run over all 353 recordings
python -m preprocessing.visualize    # plots for a few representative recordings
```

`preprocessing/pipeline.py` never reads or writes anything under
`files/` except reading the original `.wav`/`.json`/`manifest.csv` — all
generated output goes to `outputs/` (git-ignored; regenerate any time by
re-running the pipeline).

## Modules

- `audio_io.py` — loads the stereo wav, isolates Channel 0, validates
  channel count (2) and sample rate (8000 Hz), and computes basic
  pre-transformation signal stats.
- `denoise.py` — the optional noise-reduction path: a single high-pass
  Butterworth filter (cutoff 80 Hz, order 4, zero-phase). Deliberately
  simple/conservative per rules.md — it only removes very-low-frequency
  hum/rumble and never replaces the original signal, only supplements it.
- `features.py` — F0 (pYIN), RMS, and spectral (centroid/bandwidth/
  rolloff/zero-crossing-rate) extraction, all on the same frame grid
  (1024-sample / 128 ms window, 256-sample / 32 ms hop) so they line up
  in the time-series output.
- `pipeline.py` — orchestrates the above across all recordings
  (multiprocessing — this is 353 recordings × 2 signal versions, and
  pYIN is the slow step), attaches the manifest label, and writes the
  outputs described below.
- `visualize.py` — spectrogram + F0/RMS/spectral-centroid trajectory
  plots (original vs. denoised) for one example per
  (human/synthetic) × (has turn data / doesn't) combination.

## Outputs (`outputs/`, not committed)

### `outputs/features/recording_level.csv` — one row per (recording, version)

`version` is `original` or `denoised` — **not** one row per recording,
because we want the original-vs-denoised comparison (rule 11) without
guessing which one is "correct." Filter to `version == "original"` for
plain per-recording analysis.

| column | meaning |
|---|---|
| `anon_id` | recording ID, matches the `.wav`/`.json`/manifest filename |
| `label` | `human` / `synthetic`, from `manifest.csv` (source of truth) |
| `split` | `train` / `val`, from `manifest.csv` |
| `channel` | always `0` (target speaker) |
| `version` | `original` or `denoised` |
| `has_turn_data` | `False` for the 95 recordings whose JSON turn file is empty |
| `duration_s`, `sample_rate`, `n_samples` | from the raw wav |
| `raw_peak_amplitude`, `raw_rms`, `raw_silence_proportion` | computed on the untransformed signal, before any feature extraction, as a sanity check that the file loaded correctly |
| `mean_f0` / `std_f0` / `min_f0` / `max_f0` / `range_f0` | F0 (Hz) summary over voiced frames only (NaN frames excluded, never replaced) |
| `voiced_proportion` | fraction of frames pYIN judged voiced |
| `mean_rms` … `range_rms` | RMS energy trajectory summary |
| `mean_spectral_centroid` … `range_spectral_rolloff` | spectral shape summaries |
| `mean_zcr` … `range_zcr` | zero-crossing-rate summary |

### `outputs/features/timeseries.csv` — one row per (recording, version, frame)

| column | meaning |
|---|---|
| `anon_id`, `version` | as above |
| `time` | seconds from the start of the recording |
| `f0` | Hz, or empty/NaN for an unvoiced/unreliable frame |
| `voiced_flag` | pYIN's voiced/unvoiced decision for this frame |
| `rms`, `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `zcr` | same features as above, per frame |

Kept separate from the recording-level file because recordings have
different lengths/frame counts (rule 8).

### `outputs/reports/processing_report.csv` and `.md`

Per-recording `ok`/`failed` status with a reason for every failure, plus
counts (expected 353, processed, failed, with/without turn data) and the
exact preprocessing settings used for that run.

### `outputs/plots/*.png`

Spectrogram, F0, RMS, and spectral-centroid plots for a handful of
representative recordings (see `visualize.py`) — a visual check, not a
bulk output.

## Notes on "normalization"

`soundfile` reading PCM16 into floating point (`[-1, 1]`) is just a
format conversion, not loudness normalization — no per-recording
level/volume normalization is applied anywhere in this pipeline.
Amplitude-related summaries (`raw_rms`, `mean_rms`, `raw_peak_amplitude`,
etc.) reflect the recording's original level.
