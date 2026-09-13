"""
Selects a small, documented subset of the ASVspoof2021 LA evaluation
set (genuinely PSTN-transmitted vs. untransmitted "none" condition,
both containing bona fide AND spoofed/TTS-VC speech) and extracts only
those files from the downloaded archive — not the full 181,566-trial
evaluation set.

Selection (deterministic, label/generator-blind — chosen before looking
at any audio, using only the metadata): for each of `condition in
("none", "pstn")`:
  - N_SPEAKERS_PER_CONDITION bona fide speakers, N_UTT_PER_SPEAKER
    utterances each
  - N_SYSTEMS_PER_CONDITION spoofing systems (of the 13: A07-A19),
    N_UTT_PER_SYSTEM utterances each

"none" = the condition code ASVspoof2021 itself uses for NOT
transmitted/coded (closest to "clean digital recording"); "pstn" =
genuinely transmitted through the real Public Switched Telephone
Network (per the ASVspoof 2021 evaluation plan, Section 3: "Bona fide
and spoofed speech data ... is transmitted across either a public
switched telephone network (PSTN) or a voice over IP (VoIP) network").
These are NOT the same underlying recordings played through two paths
(verified: zero utt_id overlap between the two conditions) — this is a
speaker-matched and system-matched comparison (same 67 bona fide
speakers and same 13 attack systems appear under both conditions), not
an utterance-matched one. Documented explicitly, not glossed over.

Run with: python external_data/select_and_extract_asvspoof.py
"""
import random
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

EXTERNAL_DIR = Path(__file__).resolve().parent
KEYS_DIR = EXTERNAL_DIR / "raw" / "asvspoof2021_keys" / "keys" / "LA"
TRIAL_METADATA = KEYS_DIR / "CM" / "trial_metadata.txt"
ARCHIVE = EXTERNAL_DIR / "raw" / "asvspoof2021_download" / "ASVspoof2021_LA_eval.tar.gz"
RAW_OUT_DIR = EXTERNAL_DIR / "raw" / "asvspoof2021_la"
SELECTION_CSV = EXTERNAL_DIR / "manifests" / "asvspoof2021_selection.csv"

RANDOM_SEED = 42
N_SPEAKERS_PER_CONDITION = 8
N_UTT_PER_SPEAKER = 5
N_SYSTEMS_PER_CONDITION = 8
N_UTT_PER_SYSTEM = 5

COLS = ["speaker_id", "utt_id", "codec", "tx_route", "system_id", "label", "trim", "subset"]


def load_metadata() -> pd.DataFrame:
    return pd.read_csv(TRIAL_METADATA, sep=" ", names=COLS)


def select_subset(df: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)
    rows = []

    for condition in ("none", "pstn"):
        cond_df = df[df.codec == condition]

        bonafide = cond_df[cond_df.label == "bonafide"]
        speakers = sorted(bonafide.speaker_id.unique())
        chosen_speakers = rng.sample(speakers, N_SPEAKERS_PER_CONDITION)
        for spk in chosen_speakers:
            spk_utts = bonafide[bonafide.speaker_id == spk]
            chosen = spk_utts.sample(n=min(N_UTT_PER_SPEAKER, len(spk_utts)), random_state=RANDOM_SEED)
            rows.append(chosen)

        spoof = cond_df[cond_df.label == "spoof"]
        systems = sorted(spoof.system_id.unique())
        chosen_systems = rng.sample(systems, N_SYSTEMS_PER_CONDITION)
        for sysid in chosen_systems:
            sys_utts = spoof[spoof.system_id == sysid]
            chosen = sys_utts.sample(n=min(N_UTT_PER_SYSTEM, len(sys_utts)), random_state=RANDOM_SEED)
            rows.append(chosen)

    return pd.concat(rows, ignore_index=True)


def extract_selected(selection: pd.DataFrame):
    RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    members = [f"ASVspoof2021_LA_eval/flac/{utt_id}.flac" for utt_id in selection["utt_id"]]

    print(f"Extracting {len(members)} files from {ARCHIVE.name} (this reads through the archive "
          f"sequentially; may take a few minutes even though we only keep a small subset)...")
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        remaining = set(members)
        for member in tf:
            if member.name in remaining:
                tf.extract(member, path=RAW_OUT_DIR, filter="data")
                remaining.discard(member.name)
                if not remaining:
                    break
        if remaining:
            print(f"WARNING: {len(remaining)} requested files were not found in the archive: {remaining}")

    # flatten: files land in RAW_OUT_DIR/ASVspoof2021_LA_eval/flac/*.flac
    nested = RAW_OUT_DIR / "ASVspoof2021_LA_eval" / "flac"
    if nested.exists():
        for f in nested.glob("*.flac"):
            f.rename(RAW_OUT_DIR / f.name)
        (RAW_OUT_DIR / "ASVspoof2021_LA_eval" / "README.LA.txt").unlink(missing_ok=True)
        nested.rmdir()
        (RAW_OUT_DIR / "ASVspoof2021_LA_eval").rmdir()


def run():
    SELECTION_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = load_metadata()
    selection = select_subset(df)
    selection.to_csv(SELECTION_CSV, index=False)
    print(f"Selected {len(selection)} utterances -> {SELECTION_CSV}")
    print(selection.groupby(["codec", "label"]).size())

    extract_selected(selection)

    n_extracted = len(list(RAW_OUT_DIR.glob("*.flac")))
    print(f"Extracted {n_extracted} FLAC files to {RAW_OUT_DIR}")
    return selection


if __name__ == "__main__":
    run()
