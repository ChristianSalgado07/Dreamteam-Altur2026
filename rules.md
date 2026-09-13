# Project Rules — Human vs. Synthetic Speech Analysis

## 1. Project Scope

This project analyzes differences between **human speech and synthetic/AI-generated speech**.

The current phase is:

> **Data understanding, preprocessing, feature extraction, and exploratory analysis.**

The current phase is **NOT classification**.

Do not build predictive models unless explicitly requested.

The project should first identify which acoustic, temporal, conversational, and linguistic features are interesting.

---

## 2. Repository

The project is being developed on:

`https://github.com/ChristianSalgado07/Dreamteam-Altur2026/tree/tam`

The working branch is:

`tam`

Do not create a different repository or branch unless explicitly instructed.

Do not modify unrelated project work.

---

## 3. Target Speaker

Every `.wav` recording contains two channels:

```text
Channel 0 → TARGET SPEAKER
Channel 1 → OTHER SPEAKER / CONVERSATIONAL CONTEXT
```

**Channel 0 is the only speaker we are primarily evaluating.**

All detailed acoustic and NLP analysis should focus on Channel 0.

Channel 1 should mainly be used as conversational context.

Never assume that Channel 0 is human or synthetic. The dataset label determines this.

---

## 4. Preserve the Raw Dataset

Never modify, overwrite, rename, or delete the original:

* `.wav` files
* `.json` files
* CSV files

Always create derived data separately.

Preserve the original two-channel audio.

Do not permanently convert the raw recordings to mono.

---

## 5. Audio Preprocessing and Noise Reduction

Noise reduction may be useful, but it must be treated as an **optional preprocessing experiment**, not as an automatic requirement.

Possible sources of unwanted noise include:

* microphone hiss
* background conversations
* fan or AC noise
* electrical hum
* static
* low-frequency room noise

Noise can affect measurements such as:

* RMS / energy
* spectral centroid
* spectral bandwidth
* spectrograms
* pitch/F0 detection

### Important rule

**Never replace the original audio with a noise-reduced version.**

Instead:

```text
Original two-channel audio
        │
        ├──→ Original Channel 0
        │        ↓
        │   Feature extraction
        │
        └──→ Optional noise reduction
                 ↓
            Feature extraction
```

Keep both versions available for comparison.

The purpose of noise reduction is to determine whether environmental noise is interfering with the measurements.

### Do not over-denoise

Do not aggressively remove frequencies or artifacts simply because they look unusual.

Some subtle spectral characteristics may actually be relevant to distinguishing human and synthetic speech.

Noise reduction should therefore be evaluated by comparing:

* original audio
* noise-reduced audio

If noise reduction substantially changes an important feature, investigate why before deciding which version to use.

Prefer simple, interpretable methods such as:

* `scipy.signal`
* `librosa`

A dedicated library such as `noisereduce` may be tested if useful, but do not treat it as a black-box solution.

Document:

* method used
* parameters
* affected channels
* whether the audio was filtered
* how the results changed

---

## 6. Use the JSON Metadata

Each recording may have a companion `.json` file containing conversation turns with:

* channel
* start time
* end time

Example:

```json
{
  "turns": [
    {"channel": 1, "start": 0.34, "end": 3.24},
    {"channel": 1, "start": 3.62, "end": 7.48},
    {"channel": 0, "start": 7.80, "end": 12.15},
    {"channel": 1, "start": 12.40, "end": 14.02}
  ]
}
```

Use these turns to segment each channel's speech for temporal, conversational, and NLP analysis (e.g. turn duration, pauses, response latency, speaking-time ratio), rather than analyzing the raw audio as one continuous, unsegmented stream.

---

## 7. Dataset Labels

A CSV file provides the `human` / `synthetic` label for each recording.

**The CSV label is the source of truth.**

Never assume Channel 0 (or Channel 1) is human or synthetic based on the channel number alone — always resolve the label from the CSV.

---

## 8. Research Dimensions

The project investigates three core dimensions of the target speaker (Channel 0):

### Acoustic

How the target speaker sounds:

* pitch / F0
* pitch variability
* frequency characteristics
* STFT
* spectrograms
* spectral centroid
* spectral bandwidth
* spectral rolloff
* RMS / energy
* volume dynamics
* rhythm

### Temporal / Conversational

How the target speaker interacts:

* speaking duration
* turn duration
* pauses
* response latency
* speaking rate
* turn-taking
* speaking-time ratio
* conversational rhythm

Channel 1 should mainly provide conversational context for this dimension.

### NLP / Linguistic

What the target speaker says and how they use language:

* vocabulary
* lexical diversity
* sentence length
* repetition
* n-grams
* fillers
* hesitation
* POS patterns
* pronouns
* linguistic structure
* changes between turns

NLP is a core part of this project, not an optional addition.

---

## 9. Coding Style

Keep the implementation:

* simple
* readable
* modular
* beginner-friendly
* interpretable

Prefer existing project code when it already works.

Do not refactor unrelated code.

Do not introduce unnecessary frameworks or complicated abstractions.

Before implementing anything, inspect the repository and understand the existing structure.

---

## 10. Important Research Principle

Do not start by asking:

> "How can we classify human vs. synthetic speech?"

Start by asking:

> "What measurable acoustic, temporal, conversational, and linguistic patterns exist in the data?"

Understand the data first. Model later.

Do not add machine-learning classification, neural networks, embeddings, or model-training pipelines unless explicitly requested.
