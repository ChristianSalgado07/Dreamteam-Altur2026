"""
Builds the domain-matched manifest from the ASVspoof2021 LA subset
selected by select_and_extract_asvspoof.py. Adds one column beyond the
existing external_manifest.csv schema: `channel_condition` (clean /
telephone), the variable this whole experiment is designed to isolate.

Run with: python external_data/build_domain_matched_manifest.py
  (after select_and_extract_asvspoof.py)
"""
import csv
from pathlib import Path

import pandas as pd
import soundfile as sf

EXTERNAL_DIR = Path(__file__).resolve().parent
SELECTION_CSV = EXTERNAL_DIR / "manifests" / "asvspoof2021_selection.csv"
RAW_DIR = EXTERNAL_DIR / "raw" / "asvspoof2021_la"
MANIFEST_CSV = EXTERNAL_DIR / "manifests" / "domain_matched_manifest.csv"

FIELDNAMES = [
    "external_id", "source_dataset", "source_split", "label", "channel_condition",
    "speaker_id", "voice_id", "generator_id", "generator_type",
    "duration_s", "sample_rate", "channels", "original_path", "processed_path",
]

CODEC_TO_CONDITION = {"none": "clean", "pstn": "telephone"}


def run():
    selection = pd.read_csv(SELECTION_CSV)
    rows = []
    missing = []
    for _, r in selection.iterrows():
        path = RAW_DIR / f"{r['utt_id']}.flac"
        if not path.exists():
            missing.append(r["utt_id"])
            continue
        info = sf.info(str(path))
        label = "human" if r["label"] == "bonafide" else "synthetic"
        rows.append({
            "external_id": r["utt_id"],
            "source_dataset": "asvspoof2021_LA_eval",
            "source_split": "eval",
            "label": label,
            "channel_condition": CODEC_TO_CONDITION[r["codec"]],
            "speaker_id": r["speaker_id"] if label == "human" else "n/a",
            "voice_id": r["speaker_id"] if label == "human" else r["system_id"],
            "generator_id": "n/a" if label == "human" else r["system_id"],
            "generator_type": "human" if label == "human" else "tts_or_vc_asvspoof2019_attack",
            "sample_rate": info.samplerate, "channels": info.channels,
            "duration_s": round(info.frames / info.samplerate, 3),
            "original_path": str(path.relative_to(EXTERNAL_DIR)),
            "processed_path": "",
        })

    with open(MANIFEST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {MANIFEST_CSV}")
    if missing:
        print(f"WARNING: {len(missing)} selected utterances had no extracted file: {missing}")
    df = pd.DataFrame(rows)
    print(df.groupby(["channel_condition", "label"]).size())
    print(f"unique human speakers: {df.loc[df.label=='human','speaker_id'].nunique()}")
    print(f"unique synthetic systems: {df.loc[df.label=='synthetic','generator_id'].nunique()}")
    return rows


if __name__ == "__main__":
    run()
