# Dreamteam-Altur2026

Human-vs-synthetic voice detection for phone-call recordings, built in two
layers:

1. **Exploratory analysis** (`preprocessing/`, `temporal/`, `nlp/`,
   `modeling/`) — acoustic, conversational, and linguistic feature
   extraction and classical ML baselines (Logistic Regression / Random
   Forest / SVM) over the project's labeled call dataset. This is the
   "understand the data first" phase described in `rules.md`.
2. **Live real-time detector** (`app/`) — a small CNN that scores 3-second
   windows of audio in real time (browser microphone or an existing
   recording) and displays a smoothed, hysteresis-gated verdict in a
   FastAPI web demo.

This README covers both, plus the chronological history of how the live
detector went from a first working prototype to its current state.

---

## Repository structure

| Path | What it is |
|---|---|
| `files/` | The labeled dataset: `csv/manifest.csv` (anon_id, label, split, duration), `audio/*.wav` (2-channel, Channel 0 = target speaker), `turns/turns/*.json` (turn-level timing for 258/353 recordings). Never modified in place. |
| `external_data/` | External validation audio (OpenSLR human, gTTS/espeak synthetic, ASVspoof2021-LA) — see `external_data/README.md`. |
| `preprocessing/` | Channel-0 signal loading, optional denoising, acoustic feature extraction (RMS, spectral shape, F0, MFCCs). See `preprocessing/README.md`. |
| `temporal/` | Turn-taking / conversational features (turn duration, pauses, response latency) from the JSON turn metadata. See `temporal/README.md`. |
| `nlp/` | Local ASR (faster-whisper) + linguistic features (lexical diversity, fillers, hesitation). Partial/paused — see `nlp/README.md`. |
| `modeling/` | Classical baselines (`baseline.py`, `evaluate.py`), the CNN training pipeline (`train_cnn.py` + the retraining/diagnostic scripts described below), and every diagnostic script written while debugging the live detector. See `modeling/README.md`. |
| `app/` | The live detector itself: preprocessing shared with training (`audio_utils.py`), the model (`model.py`), the decision layer (`decision.py`), the rolling live-inference engine (`live_infer.py`), speaker-change guard (`speaker_guard.py`), evidence/report generation (`evidence.py`), and the FastAPI + browser demo (`main.py`). |
| `outputs/` | Generated only — trained weights, feature CSVs, and every report referenced below. Gitignored (not in the repo); regenerate locally or ask a teammate for the files if you need them. |

---

## Quickstart: running the live demo

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You need a trained CNN checkpoint at `outputs/models/cnn_live_detector.pt`
(gitignored — train one yourself with `python -m modeling.train_cnn`, or
copy an existing one from a teammate). Then:

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/`, click **Start Detection**, grant microphone
permission. The page shows a live AI-likelihood gauge, a verdict badge
(AI likely / Human likely / Analysing), a score history sparkline, and a
running "Analysis Report" (per-speaker evidence, acoustic observations,
timeline). Clicking **Stop** releases the microphone; nothing runs on the
server until Start is clicked.

Command-line alternative (no browser): `python -m app.live_infer --file
files/audio/<anon_id>.wav` or `python -m app.live_infer` for the OS
microphone via `sounddevice`.

---

## How the live detector works

**Pipeline** (identical for a live mic session and a file, via the same
`RollingDetector`):

```
Browser mic (getUserMedia, ScriptProcessorNode 4096 samples/callback)
  → raw Float32 PCM over a WebSocket (/ws/mic), no encoding, no batching
  → backend: resample to 8kHz (app/audio_utils.py) if needed
  → 3-second / 50%-overlap rolling window (RollingDetector, app/live_infer.py)
  → silence gate (raw-buffer RMS < 0.01 → treated as silence, CNN not run)
  → per-window RMS normalization (preprocessing/normalize.py)
  → log-Mel spectrogram (40 mels, librosa)
  → CNN forward pass → sigmoid → raw AI-likelihood in [0,1]
  → DecisionEngine (app/decision.py): 5-window smoothing + hysteresis
  → JSON result over the WebSocket → browser renders gauge/badge/report
```

**Model**: a small CNN (3 conv blocks + global-average-pool + linear,
~23.6K params, `app/model.py`) trained on `files/` (353 recordings,
Channel 0 only, existing train/val split, never re-randomized) via
`modeling/train_cnn.py`. Label convention: `1 = synthetic/AI`, `0 =
human` — the raw score is always P(synthetic), never inverted.

**Decision thresholds** (`app/decision.py`, the single place to change
them):
```python
AI_THRESHOLD = 0.90        # smoothed score >= this → "AI likely"
HUMAN_THRESHOLD = 0.10     # smoothed score <= this → "Human likely"
SMOOTHING_WINDOW = 5        # raw window scores averaged
CONSECUTIVE_REQUIRED = 2    # windows needed in agreement before the label flips
SILENCE_RMS_THRESHOLD = 0.01
```
Deliberately conservative: a wide 0.10–0.90 band never forces a verdict.
The displayed label is also **sticky** — once "AI likely" or "Human
likely" is confirmed for the current speaker, an ambiguous stretch keeps
showing it rather than reverting to a neutral state; "Analysing" only
appears before any verdict has been confirmed (session start, or right
after a confirmed speaker change resets evidence for the new speaker).
Silence is a **cooldown, not a reset** — it never touches the smoothing
history or the confirmed label, and evidence resumes exactly where it
left off once speech returns.

**Speaker-change guard** (`app/speaker_guard.py`): an F0/spectral-shape
heuristic that pauses prediction for a couple of windows when a
different speaker appears to start talking, so a new speaker's evidence
never contaminates the previous one's running score.

**Known limitation, stated in the UI's own disclaimer**: internal
validation ROC-AUC is ~1.00 but external validation (OpenSLR + gTTS/
espeak, `external_data/`) is ~0.53 — near chance. This is a prototype
that analyzes acoustic patterns in real time, not a scientifically
validated universal AI-voice detector.

---

## How we got here

Roughly chronological; full detail for each item lives in the
`outputs/reports/*.md` file named (generated locally, gitignored — this
README summarizes what each one found).

1. **Exploratory baseline** (`modeling/baseline.py`, `evaluate.py`,
   `outputs/reports/modeling_evaluation_report.md`) — Logistic
   Regression / RF / SVM on acoustic + MFCC + temporal features. Scored
   near-perfect ROC-AUC, which triggered a thorough leakage/confound
   investigation (feature-family ablation, denoising comparison,
   Channel-1 cross-check) — separation looked real but possibly
   pipeline-specific, not yet a confirmed generalizable detector.
2. **CNN live-detector prototype** (`modeling/train_cnn.py`,
   `outputs/reports/cnn_training_report.md`) — a log-Mel spectrogram CNN
   trained on 3-second windows, wired into a live FastAPI demo
   (`app/main.py`) with a browser mic path and a decision/smoothing
   layer. Internal ROC-AUC ~1.00, external ROC-AUC ~0.53.
3. **Hallucination diagnostic** (`modeling/diagnose_cnn_hallucination.py`,
   `cnn_hallucination_diagnostics.md`) — root-caused false "AI" scores on
   real human speech to (a) ~53% of training windows being near-silent
   and (b) an F0-register correlation with the synthetic class.
4. **Speech-only retraining experiment** (`train_cnn_speech_only.py`) —
   tried excluding near-silent windows entirely. Made things worse
   (recording accuracy 1.00 → 0.54) — a calibration collapse, documented
   and **not shipped**.
5. **Speech-replaced retraining experiment**
   (`train_cnn_speech_replaced.py`, `cnn_speech_replaced_comparison.md`)
   — replaced (not removed) near-silent training windows with
   same-recording speech, holding window count fixed. External ROC-AUC
   0.53 → 0.79; false positives on the known-problem human recordings
   dropped from 0.98 to 0.03–0.33.
6. **Silence-diagnostic + live-pipeline comparison**
   (`diagnose_cnn_silence.py`, `compare_live_pipeline.py`,
   `live_detector_comparison.md`) — confirmed the real production
   `RollingDetector` (silence gate active) already neutralizes the
   remaining near-silence miscalibration; all 5 promotion criteria met.
7. **Promoted `cnn_speech_replaced.pt` to production**
   (`outputs/models/cnn_live_detector.pt`), baseline preserved at
   `cnn_baseline.pt`, `CNN_MODEL_PATH` env override added for rollback —
   see `final_live_demo_validation.md`.
8. **AI-threshold diagnostic** (`diagnose_ai_threshold.py`,
   `ai_threshold_diagnostic.md`) — after promotion, checked whether
   reported AI under-detection was a threshold problem. Synthetic scores
   were consistently >0.90 on the recordings tested, so — per the task's
   own explicit gate — **no threshold change was made** at that time.
9. **`mich_demo` branch comparison** (`mich_demo_comparison.md`) —
   inspected a teammate's alternate XGBoost/hand-crafted-feature
   approach. No trained checkpoint existed in that branch's git history;
   running its own feature extractor on shared files showed no evidence
   it was a genuinely stronger model. Nothing copied over.
10. **Live-mic-specific diagnostics** — three separate investigations
    once the symptom was narrowed to "works on file upload, not on live
    mic":
    - `diagnose_live_vs_file_pipeline.py` — found the mic path's
      per-chunk stateless resampling (browser audio arrives at 44.1/48kHz
      in small pieces, file-mode audio is already 8kHz and never
      resamples) causes occasional silence-gate flips and modest score
      drift — real, but modest.
    - `diagnose_live_latency.py` (`live_stream_latency.md`) — measured
      actual transport/backend latency through the real running server:
      ~2ms network, ~120–130ms inference. **Not the bottleneck.**
    - `mic_processing_ab_test.md` — added a `?micraw=1` toggle to test
      whether Chrome's default echo-cancellation/noise-suppression/AGC
      alters live scores. Instrumentation shipped; the physical mic+
      speaker test itself needs a real machine (this was built/verified
      in a sandboxed environment with no microphone) and is still
      pending a run.
11. **Stricter, sticky decision thresholds** — raised `AI_THRESHOLD`/
    `HUMAN_THRESHOLD` from 0.65/0.35 to 0.90/0.10 (single change point,
    `app/decision.py`), then made the displayed label sticky (an
    ambiguous stretch no longer reverts a confirmed verdict) and renamed
    the neutral state "Analysing".
12. **UI redesign** — restyled the FastAPI demo page to match the
    Altur frontend's design system (`main` branch: Inter font, ink/
    surface/line dark palette, card layout, a semi-circular SVG gauge
    replacing the old flat progress bar) and removed the "Demo / File
    Mode" fallback UI (and its now-unused `/mode`/`/score` backend
    endpoints) — `python -m app.live_infer --file` and the
    `modeling/diagnose_*` scripts already cover that use case.
13. **Committed to `tam`** — 435 previously-untracked files (all of the
    above) added in one commit; raw audio, model checkpoints, and
    `outputs/` remain gitignored (kept local-only for now).

---

## What's NOT done yet

- The `?micraw=1` mic-processing A/B test needs an actual physical
  microphone/speaker run (see `mic_processing_ab_test.md` for the exact
  steps) — not yet executed.
- External validation (~0.53 ROC-AUC) means the model should not be
  presented as a validated universal detector — see
  `cnn_training_report.md`.
- `outputs/` (trained weights + every report above) is gitignored. If a
  teammate clones fresh, they will not have a model checkpoint to load
  or these reports to read until one is trained/shared separately.
