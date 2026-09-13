"""
Generates the controlled ElevenLabs dataset for the AI-voice-fingerprint
stage (see rules.md and outputs/reports/elevenlabs_fingerprint_report.md).

This is a NEW, additive data-collection step. It does not touch, resample,
or relabel anything under files/, external_data/raw/openslr61_human/,
external_data/raw/synthetic_gtts/, external_data/raw/synthetic_espeak/, or
external_data/raw/asvspoof2021_la/ — those remain exactly as produced by
prior experiments.

Text-matching design (task requirement: "use the SAME text content for
human recordings and ElevenLabs recordings where possible"):
  The human control group for this stage reuses the EXISTING 40 OpenSLR
  SLR61 (Argentinian Spanish) clips already in
  external_data/raw/openslr61_human/ (see external_data/README.md) rather
  than downloading anything new. Their exact transcripts are looked up
  from the already-present line_index.tsv (the full 1818-line index was
  kept even though only 40 audio files were extracted). This script
  selects a deterministic subset of those exact sentences and generates
  ElevenLabs speech for EACH ONE, so every ElevenLabs clip has a matched
  human recording of the identical text (by a different speaker) — a
  genuine same-text comparison, not merely same-language.

Voice selection: fetched live from GET /v1/voices (never hard-coded), so
the manifest always reflects the voice_ids actually used, not assumed
ones. Selection is deterministic given the same account's voice list:
voices are sorted by voice_id for a stable order; if any voice's labels
mention a Spanish/Latin-American/Mexican accent or language, those are
preferred first (documented per-voice in the generation manifest via the
`labels_used_for_selection` column), otherwise the script falls back to
the first N_VOICES premade voices in sorted order and records that the
fallback path was used.

Model: eleven_multilingual_v2 (required for non-English text; ElevenLabs
voices are timbre models, not language-locked, so the multilingual model
renders the same voice identity in Spanish).

Required environment variable:
  ELEVENLABS_API_KEY   -- your ElevenLabs API key. NEVER hard-code this;
                          export it in your own shell before running:
                              export ELEVENLABS_API_KEY=sk_...
Optional environment variables:
  ELEVENLABS_VOICE_IDS -- comma-separated voice_id override, e.g.
                          "21m00Tcm4TlvDq8ikWAM,AZnzlk1XvdvUeBnXmlld"
                          Skips the /v1/voices selection logic above and
                          uses exactly these voices (still validated to
                          exist via GET /v1/voices/{id} before use).
  ELEVENLABS_N_VOICES   -- override N_VOICES (default 8)
  ELEVENLABS_N_SENTENCES -- override N_SENTENCES (default 12)

If ELEVENLABS_API_KEY is not set, this script prints the requirement and
exits without making any network call, generating any placeholder audio,
or writing any manifest row -- per the project rule to never fabricate
data when a required external credential/service is unavailable.

Run with: python external_data/generate_elevenlabs.py
  (after external_data/raw/openslr61_human/ already exists, which it
  does from the earlier external-validation experiment)
"""
import csv
import os
import re
import sys
import time
from pathlib import Path

import requests

EXTERNAL_DIR = Path(__file__).resolve().parent
HUMAN_DIR = EXTERNAL_DIR / "raw" / "openslr61_human"
LINE_INDEX_TSV = HUMAN_DIR / "line_index.tsv"
ELEVENLABS_DIR = EXTERNAL_DIR / "elevenlabs"
RAW_OUT_DIR = ELEVENLABS_DIR / "raw"
GENERATION_MANIFEST_CSV = ELEVENLABS_DIR / "generation_manifest.csv"

API_BASE = "https://api.elevenlabs.io/v1"
MODEL_ID = "eleven_multilingual_v2"
OUTPUT_FORMAT = "mp3_44100_128"  # ElevenLabs default-quality format; soundfile/libsndfile decodes mp3 the same way the existing gTTS clips already do in this project

N_VOICES = int(os.environ.get("ELEVENLABS_N_VOICES", "8"))
N_SENTENCES = int(os.environ.get("ELEVENLABS_N_SENTENCES", "12"))

# Accent/language hint keywords used ONLY to prefer voices when a voice's
# own ElevenLabs-provided labels mention them -- never inferred, never
# assumed of a voice that doesn't declare it.
SPANISH_HINTS = ("spanish", "latin", "mexic", "es-mx", "es_mx", "argentin", "latam")


def select_matched_sentences(n: int) -> list:
    """
    Returns up to `n` (utt_id, text) pairs drawn from the EXACT transcripts
    of the 40 existing external_data/raw/openslr61_human/*.wav clips (see
    module docstring). Deterministic: sorted by utt_id, first n taken, one
    duplicate text (if any) is kept only once.
    """
    human_ids = {p.stem for p in sorted(HUMAN_DIR.glob("*.wav"))}
    if not human_ids:
        raise FileNotFoundError(
            f"No existing human clips found under {HUMAN_DIR} -- this "
            "script reuses the external-validation experiment's OpenSLR "
            "subset and expects it to already exist."
        )
    pairs = []
    seen_texts = set()
    with open(LINE_INDEX_TSV, encoding="utf-8") as fh:
        rows = []
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2 and parts[0] in human_ids:
                rows.append((parts[0], parts[1]))
    for utt_id, text in sorted(rows):
        if text in seen_texts:
            continue
        seen_texts.add(text)
        pairs.append((utt_id, text))
    if len(pairs) < n:
        raise ValueError(
            f"Only {len(pairs)} unique matched transcripts available under "
            f"{HUMAN_DIR}, need {n}. Lower ELEVENLABS_N_SENTENCES or add "
            "more human clips first."
        )
    return pairs[:n]


def fetch_voices(api_key: str) -> list:
    resp = requests.get(f"{API_BASE}/voices", headers={"xi-api-key": api_key}, timeout=30)
    resp.raise_for_status()
    return resp.json()["voices"]


def select_voices(voices: list, n: int) -> list:
    """
    Returns up to n voice dicts. Prefers voices whose labels mention a
    Spanish/Latin-American accent or language (see SPANISH_HINTS);
    otherwise falls back to the first n by voice_id. Documents which path
    was used per voice via the returned 'selection_reason' key.
    """
    def label_text(v):
        return " ".join(str(x).lower() for x in (v.get("labels") or {}).values())

    hinted = [v for v in voices if any(h in label_text(v) for h in SPANISH_HINTS)]
    hinted_sorted = sorted(hinted, key=lambda v: v["voice_id"])
    for v in hinted_sorted:
        v["selection_reason"] = "label_matched_spanish_hint"

    remaining_needed = n - len(hinted_sorted)
    selected = hinted_sorted[:n]
    if remaining_needed > 0:
        already_ids = {v["voice_id"] for v in selected}
        fallback_pool = sorted(
            [v for v in voices if v["voice_id"] not in already_ids],
            key=lambda v: v["voice_id"],
        )
        for v in fallback_pool[:remaining_needed]:
            v["selection_reason"] = "fallback_no_spanish_label_found"
            selected.append(v)
    return selected[:n]


def run():
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print(
            "ELEVENLABS_API_KEY is not set in this environment.\n\n"
            "This script requires a real ElevenLabs API key to generate\n"
            "the controlled synthetic dataset for the AI-voice-fingerprint\n"
            "stage. Per project rules, no ElevenLabs audio is fabricated\n"
            "or substituted when this credential is unavailable.\n\n"
            "To proceed:\n"
            "  1. Obtain an API key from https://elevenlabs.io (your own\n"
            "     account -- never share/commit this key).\n"
            "  2. In your shell (NOT in this chat): \n"
            "         export ELEVENLABS_API_KEY=sk_...\n"
            "  3. Re-run: python external_data/generate_elevenlabs.py\n\n"
            "Optional overrides: ELEVENLABS_VOICE_IDS, ELEVENLABS_N_VOICES,\n"
            "ELEVENLABS_N_SENTENCES (see this script's docstring).\n"
        )
        sys.exit(1)

    ELEVENLABS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)

    sentences = select_matched_sentences(N_SENTENCES)
    print(f"Using {len(sentences)} matched sentences (same text as existing human clips):")
    for utt_id, text in sentences:
        print(f"  [{utt_id}] {text}")

    voice_id_override = os.environ.get("ELEVENLABS_VOICE_IDS")
    if voice_id_override:
        ids = [v.strip() for v in voice_id_override.split(",") if v.strip()]
        voices = []
        for vid in ids:
            resp = requests.get(f"{API_BASE}/voices/{vid}", headers={"xi-api-key": api_key}, timeout=30)
            resp.raise_for_status()
            v = resp.json()
            v["selection_reason"] = "explicit_env_override"
            voices.append(v)
        print(f"Using {len(voices)} explicitly-specified voice(s) from ELEVENLABS_VOICE_IDS.")
    else:
        all_voices = fetch_voices(api_key)
        voices = select_voices(all_voices, N_VOICES)
        print(f"Selected {len(voices)}/{len(all_voices)} voices from the account's voice list.")

    if not voices:
        print("No ElevenLabs voices available on this account/API key. Stopping without generating audio.")
        sys.exit(1)

    rows = []
    for v in voices:
        voice_id = v["voice_id"]
        voice_name = re.sub(r"[^A-Za-z0-9_-]", "_", v.get("name", voice_id))
        voice_dir = RAW_OUT_DIR / voice_name
        voice_dir.mkdir(parents=True, exist_ok=True)
        for i, (utt_id, text) in enumerate(sentences):
            clip_id = f"elabs_{voice_name}_{i:03d}"
            out_path = voice_dir / f"{clip_id}.mp3"
            resp = requests.post(
                f"{API_BASE}/text-to-speech/{voice_id}",
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "text": text,
                    "model_id": MODEL_ID,
                    "output_format": OUTPUT_FORMAT,
                },
                timeout=60,
            )
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            rows.append({
                "external_id": clip_id,
                "voice_id": voice_id,
                "voice_name": v.get("name", voice_id),
                "selection_reason": v.get("selection_reason", ""),
                "text": text,
                "matched_human_utt_id": utt_id,
                "model_id": MODEL_ID,
                "output_format": OUTPUT_FORMAT,
                "raw_file": str(out_path.relative_to(EXTERNAL_DIR)),
            })
            print(f"generated {clip_id} (voice={v.get('name', voice_id)})")
            time.sleep(0.2)  # gentle pacing, not a rate-limit workaround for anything adversarial

    with open(GENERATION_MANIFEST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {GENERATION_MANIFEST_CSV}")
    print(f"Voices used: {sorted(set(r['voice_name'] for r in rows))}")
    print(f"Sentences used: {len(sentences)} (matched to existing human transcripts)")


if __name__ == "__main__":
    run()
