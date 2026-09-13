"""
Builds the unified external manifest from the two raw sources (see
external_data/README.md for full provenance/licensing). Labels are
taken directly from which source/script produced each clip — never
inferred from audio content or filename patterns beyond the documented,
verified ID extraction described below.

`processed_path` is left empty here (filled in later by
modeling/external_validation.py's audio-adapter step, which re-writes
this same manifest after producing the 8kHz mono versions).

No external train/dev/test split is created: this experiment makes NO
decisions based on external results (no tuning, no feature/model
selection), so the entire external set is used as a single strict
holdout — see external_data/README.md.

Run with: python external_data/build_manifest.py
"""
import csv
import re
from pathlib import Path

import soundfile as sf

EXTERNAL_DIR = Path(__file__).resolve().parent
RAW_DIR = EXTERNAL_DIR / "raw"
MANIFEST_DIR = EXTERNAL_DIR / "manifests"
SYNTH_GEN_MANIFEST = MANIFEST_DIR / "synthetic_generation_manifest.csv"
EXTERNAL_MANIFEST = MANIFEST_DIR / "external_manifest.csv"

FIELDNAMES = [
    "external_id", "source_dataset", "source_split", "label",
    "speaker_id", "voice_id", "generator_id", "generator_type",
    "duration_s", "sample_rate", "channels", "original_path", "processed_path",
]


def audio_info(path: Path) -> dict:
    info = sf.info(str(path))
    return {"duration_s": round(info.frames / info.samplerate, 3),
            "sample_rate": info.samplerate, "channels": info.channels}


def build_human_rows() -> list:
    human_dir = RAW_DIR / "openslr61_human"
    rows = []
    for wav_path in sorted(human_dir.glob("*.wav")):
        m = re.match(r"(arm_\d+)_(\d+)", wav_path.stem)
        speaker_id = m.group(1) if m else "unknown"
        info = audio_info(wav_path)
        rows.append({
            "external_id": wav_path.stem,
            "source_dataset": "openslr61_argentinian_spanish",
            "source_split": "n/a",  # no official train/test split published
            "label": "human",
            "speaker_id": speaker_id,
            "voice_id": speaker_id,
            "generator_id": "n/a",
            "generator_type": "human",
            "original_path": str(wav_path.relative_to(EXTERNAL_DIR)),
            "processed_path": "",
            **info,
        })
    return rows


def build_synthetic_rows() -> list:
    rows = []
    generator_type = {"gtts_google": "web_tts", "espeak_ng": "formant_synthesis"}
    with open(SYNTH_GEN_MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            raw_path = Path(r["raw_file"])
            info = audio_info(raw_path)
            rows.append({
                "external_id": r["external_id"],
                "source_dataset": "self_generated_open_tts",
                "source_split": "n/a",
                "label": "synthetic",
                "speaker_id": "n/a",
                "voice_id": f"{r['generator_id']}_es_default",
                "generator_id": r["generator_id"],
                "generator_type": generator_type.get(r["generator_id"], "unknown"),
                "original_path": str(raw_path.relative_to(EXTERNAL_DIR)),
                "processed_path": "",
                **info,
            })
    return rows


def run():
    rows = build_human_rows() + build_synthetic_rows()
    with open(EXTERNAL_MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {EXTERNAL_MANIFEST}")
    n_human = sum(1 for r in rows if r["label"] == "human")
    n_synth = sum(1 for r in rows if r["label"] == "synthetic")
    n_speakers = len({r["speaker_id"] for r in rows if r["label"] == "human"})
    n_generators = len({r["generator_id"] for r in rows if r["label"] == "synthetic"})
    print(f"human={n_human}, synthetic={n_synth}, unique human speakers={n_speakers}, unique generators={n_generators}")
    return rows


if __name__ == "__main__":
    run()
