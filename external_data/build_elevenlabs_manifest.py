"""
Builds external_data/elevenlabs/manifest.csv -- the unified manifest for
the AI-voice-fingerprint stage, joining:

  1. The human control group: the EXISTING 40 OpenSLR SLR61 (Argentinian
     Spanish) clips at external_data/raw/openslr61_human/ (untouched,
     reused as-is -- no new download, no copy of the audio itself; this
     manifest just points at the existing files via a relative path).
  2. The ElevenLabs synthetic group: external_data/elevenlabs/generation_manifest.csv,
     produced by external_data/generate_elevenlabs.py (requires
     ELEVENLABS_API_KEY; see that script).

`text_id` = the OpenSLR utt_id whose transcript a row's text matches
exactly (for BOTH human rows, where it's the row's own id, and
ElevenLabs rows, where it's the human utt_id whose text was reused --
see generate_elevenlabs.py). This makes same-text comparison a simple
groupby on `text_id` rather than a fuzzy text match.

If external_data/elevenlabs/generation_manifest.csv does not exist yet
(ElevenLabs generation blocked on credentials), this script stops with a
clear message and writes NOTHING -- it does not fabricate a
human-only or placeholder manifest.

Run with: python external_data/build_elevenlabs_manifest.py
  (after external_data/generate_elevenlabs.py has produced
   external_data/elevenlabs/generation_manifest.csv)
"""
import csv
import re
import sys
from pathlib import Path

import soundfile as sf

EXTERNAL_DIR = Path(__file__).resolve().parent
HUMAN_DIR = EXTERNAL_DIR / "raw" / "openslr61_human"
LINE_INDEX_TSV = HUMAN_DIR / "line_index.tsv"
ELEVENLABS_DIR = EXTERNAL_DIR / "elevenlabs"
GENERATION_MANIFEST_CSV = ELEVENLABS_DIR / "generation_manifest.csv"
UNIFIED_MANIFEST_CSV = ELEVENLABS_DIR / "manifest.csv"

FIELDNAMES = [
    "anon_id", "source", "label", "speaker_id", "voice_id", "generator",
    "language", "sample_rate", "duration_s", "channels", "text_id", "text",
    "preprocessing_variant", "original_path", "processed_path",
]


def audio_info(path: Path) -> dict:
    info = sf.info(str(path))
    return {"duration_s": round(info.frames / info.samplerate, 3),
            "sample_rate": info.samplerate, "channels": info.channels}


def build_human_rows() -> list:
    id_to_text = {}
    with open(LINE_INDEX_TSV, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                id_to_text[parts[0]] = parts[1]

    rows = []
    for wav_path in sorted(HUMAN_DIR.glob("*.wav")):
        m = re.match(r"(arm_\d+)_(\d+)", wav_path.stem)
        speaker_id = m.group(1) if m else "unknown"
        info = audio_info(wav_path)
        rows.append({
            "anon_id": wav_path.stem,
            "source": "openslr61_argentinian_spanish",
            "label": "human",
            "speaker_id": speaker_id,
            "voice_id": "n/a",
            "generator": "n/a",
            "language": "es_ar",
            "text_id": wav_path.stem,
            "text": id_to_text.get(wav_path.stem, ""),
            "preprocessing_variant": "",
            "original_path": str((HUMAN_DIR / f"{wav_path.stem}.wav").relative_to(EXTERNAL_DIR)),
            "processed_path": "",
            **info,
        })
    return rows


def build_elevenlabs_rows() -> list:
    rows = []
    with open(GENERATION_MANIFEST_CSV, newline="") as f:
        for r in csv.DictReader(f):
            raw_path = EXTERNAL_DIR / r["raw_file"]
            info = audio_info(raw_path)
            rows.append({
                "anon_id": r["external_id"],
                "source": "elevenlabs_self_generated",
                "label": "synthetic",
                "speaker_id": "n/a",
                "voice_id": r["voice_id"],
                "generator": "ElevenLabs",
                "language": "es",  # requested language of generation; see report for the es_ar (human) vs es (requested, accent not guaranteed) dialect caveat
                "text_id": r["matched_human_utt_id"],
                "text": r["text"],
                "preprocessing_variant": "",
                "original_path": r["raw_file"],
                "processed_path": "",
                **info,
            })
    return rows


def run():
    if not GENERATION_MANIFEST_CSV.exists():
        print(
            f"{GENERATION_MANIFEST_CSV} does not exist yet -- ElevenLabs "
            "generation has not been run (likely blocked on "
            "ELEVENLABS_API_KEY; see external_data/generate_elevenlabs.py).\n"
            "Stopping without writing a manifest -- a human-only or "
            "placeholder manifest would misrepresent this stage's data."
        )
        sys.exit(1)

    rows = build_human_rows() + build_elevenlabs_rows()
    ELEVENLABS_DIR.mkdir(parents=True, exist_ok=True)
    with open(UNIFIED_MANIFEST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    n_human = sum(1 for r in rows if r["label"] == "human")
    n_synth = sum(1 for r in rows if r["label"] == "synthetic")
    n_voices = len({r["voice_id"] for r in rows if r["label"] == "synthetic"})
    n_speakers = len({r["speaker_id"] for r in rows if r["label"] == "human"})
    human_text_ids = {r["text_id"] for r in rows if r["label"] == "human"}
    synth_text_ids = {r["text_id"] for r in rows if r["label"] == "synthetic"}
    n_matched_texts = len(human_text_ids & synth_text_ids)

    print(f"Wrote {len(rows)} rows to {UNIFIED_MANIFEST_CSV}")
    print(f"human={n_human} ({n_speakers} speakers), synthetic={n_synth} ({n_voices} ElevenLabs voices)")
    print(f"text_ids with BOTH a human and an ElevenLabs recording: {n_matched_texts}/{len(synth_text_ids)}")


if __name__ == "__main__":
    run()
