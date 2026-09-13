"""
======================================================================
DIAGNOSTIC-ONLY SCRIPT -- NOT PRODUCTION CODE.

Written for a one-time investigation ("why is the CNN hallucinating?").
Does NOT modify the model, weights, thresholds, preprocessing, training
data, or any production module. Safe to delete after the diagnostic
report (outputs/reports/cnn_hallucination_diagnostics.md) is read.
======================================================================

Runs every numeric check requested for the hallucination diagnostic:
  - tensor shape / dtype / value-range check (train-style vs live-style)
  - log-mel distribution stats: train windows vs external (out-of-
    pipeline) windows, as a live-microphone proxy
  - human vs synthetic training log-mel distributions
  - how many of the ACTUAL windows selected for training are
    near-silent by the live silence-gate's own standard
  - raw CNN behavior on fixed known windows, no smoothing/decision layer
  - determinism (same tensor, 10 runs)
  - most-suspicious human windows (highest raw AI score) + their
    acoustic features vs a normal-window baseline
  - recording-level / window-level class balance
  - train/val split leakage check

Prints a big structured dump; also writes
outputs/reports/_diagnostic_raw_dump.json for the report-writing step.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from preprocessing.audio_io import load_channel0, MANIFEST_CSV, AUDIO_DIR
from preprocessing.features import extract_rms, extract_spectral, extract_f0
from app.audio_utils import (
    TARGET_SR, WINDOW_SAMPLES, HOP_SAMPLES, WINDOW_SECONDS, HOP_SECONDS,
    N_MELS, N_FFT, HOP_LENGTH, N_FRAMES,
    slice_windows, log_mel_spectrogram,
)
from preprocessing.normalize import normalize_rms
from app.model import load_model, VoiceCNN
from modeling.train_cnn import (
    build_windows_for_recordings, subsample_windows, MAX_WINDOWS_PER_RECORDING,
    LABEL_TO_INT,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "outputs" / "models" / "cnn_live_detector.pt"
DUMP_PATH = REPO_ROOT / "outputs" / "reports" / "_diagnostic_raw_dump.json"
EXTERNAL_DIR = REPO_ROOT / "external_data"
EXTERNAL_MANIFEST_CSV = EXTERNAL_DIR / "manifests" / "external_manifest.csv"
EXTERNAL_PROCESSED_DIR = EXTERNAL_DIR / "processed" / "original"

SILENCE_RMS_THRESHOLD = 0.01  # same constant as app/decision.py, duplicated here read-only for comparison

dump = {}


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def describe(arr, name):
    arr = np.asarray(arr, dtype=np.float64)
    d = {
        "n": int(arr.size), "shape": list(arr.shape), "dtype": str(arr.dtype),
        "mean": float(np.mean(arr)), "std": float(np.std(arr)),
        "min": float(np.min(arr)), "max": float(np.max(arr)),
        "p1": float(np.percentile(arr, 1)), "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
    print(f"{name}: n={d['n']} shape={d['shape']} dtype={d['dtype']} "
          f"mean={d['mean']:.3f} std={d['std']:.3f} min={d['min']:.3f} max={d['max']:.3f} "
          f"p1={d['p1']:.3f} p5={d['p5']:.3f} p50={d['p50']:.3f} p95={d['p95']:.3f} p99={d['p99']:.3f}")
    return d


# ----------------------------------------------------------------------
section("0. Load model + manifest")
# ----------------------------------------------------------------------
model = load_model(MODEL_PATH)
manifest = pd.read_csv(MANIFEST_CSV)
labels_by_id = dict(zip(manifest["anon_id"], manifest["label"]))
train_ids = manifest[manifest["split"] == "train"]["anon_id"].tolist()
val_ids = manifest[manifest["split"] == "val"]["anon_id"].tolist()
print(f"train={len(train_ids)} val={len(val_ids)}")
print(f"train/val anon_id overlap: {len(set(train_ids) & set(val_ids))} (must be 0)")
dump["split_overlap"] = len(set(train_ids) & set(val_ids))
dump["n_train_recordings"] = len(train_ids)
dump["n_val_recordings"] = len(val_ids)


# ----------------------------------------------------------------------
section("1. Tensor shape / dtype check: train-style vs live-style window")
# ----------------------------------------------------------------------
sample_id = val_ids[0]
y_audio, sr = load_channel0(sample_id)
train_style_windows = slice_windows(y_audio, sr)  # whole-recording normalize, THEN slice
mel_train_style = log_mel_spectrogram(train_style_windows[5])

live_buffer = y_audio[int(30 * sr):int(30 * sr) + WINDOW_SAMPLES]  # a single 3s window, as a live buffer would hold
y_norm_live, gain_info_live = normalize_rms(live_buffer)  # per-WINDOW normalize (live path)
mel_live_style = log_mel_spectrogram(y_norm_live)

print(f"WINDOW_SAMPLES={WINDOW_SAMPLES} N_MELS={N_MELS} N_FFT={N_FFT} HOP_LENGTH={HOP_LENGTH} N_FRAMES={N_FRAMES}")
print(f"train-style mel shape={mel_train_style.shape} dtype={mel_train_style.dtype}")
print(f"live-style  mel shape={mel_live_style.shape} dtype={mel_live_style.dtype}")
print(f"shapes match: {mel_train_style.shape == mel_live_style.shape}")

x_train = torch.from_numpy(mel_train_style).unsqueeze(0).unsqueeze(0)
x_live = torch.from_numpy(mel_live_style).unsqueeze(0).unsqueeze(0)
print(f"model input tensor shape: {tuple(x_train.shape)} (train-style), {tuple(x_live.shape)} (live-style)")
dump["tensor_shape"] = {"train_style": list(mel_train_style.shape), "live_style": list(mel_live_style.shape)}


# ----------------------------------------------------------------------
section("2. Does whole-recording vs per-window RMS normalization actually change the log-mel tensor?")
# ----------------------------------------------------------------------
# Same underlying 3s of raw audio, once via the TRAINING path (normalize
# applied to the WHOLE recording, then sliced) and once via the LIVE
# path (normalize applied to JUST this 3s window). log_mel_spectrogram
# uses librosa.power_to_db(mel, ref=np.max) -- ref=np.max makes the
# result invariant to any uniform positive gain on the input, so if
# these two preprocessing paths differ ONLY by a linear gain, the
# resulting tensors should be numerically identical (this is a specific,
# falsifiable, code-level hypothesis -- verified directly below, not
# assumed).
y_audio_full, _ = load_channel0(sample_id)
y_full_norm, whole_gain_info = normalize_rms(y_audio_full)  # whole-recording gain, as training's slice_windows does
window_from_whole = y_full_norm[int(30 * sr):int(30 * sr) + WINDOW_SAMPLES]
mel_from_whole_recording_gain = log_mel_spectrogram(window_from_whole)

window_raw = y_audio_full[int(30 * sr):int(30 * sr) + WINDOW_SAMPLES]
window_own_gain, own_gain_info = normalize_rms(window_raw)  # per-window gain, as live step() does
mel_from_own_window_gain = log_mel_spectrogram(window_own_gain)

max_abs_diff = float(np.max(np.abs(mel_from_whole_recording_gain - mel_from_own_window_gain)))
print(f"whole-recording gain applied to this window: {whole_gain_info['gain']:.4f}")
print(f"per-window gain applied to this SAME window: {own_gain_info['gain']:.4f}")
print(f"max abs difference in resulting log-mel tensors: {max_abs_diff:.6f} dB")
print("CONCLUSION:", "log-mel tensors are effectively IDENTICAL despite the different gain scope"
      if max_abs_diff < 0.01 else "log-mel tensors DIFFER MEANINGFULLY -- gain scope DOES matter")
dump["gain_scope_max_abs_diff_db"] = max_abs_diff
dump["whole_recording_gain"] = whole_gain_info["gain"]
dump["per_window_gain"] = own_gain_info["gain"]


# ----------------------------------------------------------------------
section("3. How many ACTUAL training windows (as selected by build_windows_for_recordings) are near-silent?")
# ----------------------------------------------------------------------
# Uses the real training window-selection code, unmodified, on a sample
# of recordings (all 71 val recordings, since that's the internally
# reported validation set) -- checks each SELECTED raw window's RMS
# (pre-normalization, matching the live silence gate's own convention)
# against the live SILENCE_RMS_THRESHOLD.
below_threshold = 0
total_windows = 0
window_raw_rms_values = []
per_class_raw_rms = {"human": [], "synthetic": []}
for anon_id in val_ids:
    y_audio, sr = load_channel0(anon_id)
    raw_windows = subsample_windows(slice_windows(y_audio, sr), MAX_WINDOWS_PER_RECORDING)
    # NOTE: slice_windows already RMS-normalized these (whole-recording
    # gain) -- to check "was this originally near-silent", we need the
    # RAW (pre-normalization) RMS of the same time range. Recompute
    # directly from the raw audio at matching sample offsets.
    y_raw, _ = load_channel0(anon_id)
    y_raw = np.asarray(y_raw, dtype=np.float64)
    # Recreate the same raw (unnormalized) windows via make_windows-equivalent slicing
    from app.audio_utils import make_windows
    unnorm_windows_all = make_windows(y_raw if sr == TARGET_SR else y_raw)  # this dataset is already 8kHz
    unnorm_selected = subsample_windows(unnorm_windows_all, MAX_WINDOWS_PER_RECORDING)
    label = labels_by_id[anon_id]
    for w in unnorm_selected:
        rms = float(np.sqrt(np.mean(w ** 2)))
        window_raw_rms_values.append(rms)
        per_class_raw_rms[label].append(rms)
        total_windows += 1
        if rms < SILENCE_RMS_THRESHOLD:
            below_threshold += 1

print(f"total selected training(val) windows checked: {total_windows}")
print(f"windows BELOW the live silence threshold ({SILENCE_RMS_THRESHOLD}): {below_threshold} "
      f"({100*below_threshold/total_windows:.1f}%)")
describe(window_raw_rms_values, "raw RMS of all selected windows")
describe(per_class_raw_rms["human"], "raw RMS of HUMAN selected windows")
describe(per_class_raw_rms["synthetic"], "raw RMS of SYNTHETIC selected windows")
dump["pct_training_windows_below_live_silence_threshold"] = 100 * below_threshold / total_windows
dump["n_training_windows_checked"] = total_windows


# ----------------------------------------------------------------------
section("4. Train-window log-mel distribution: human vs synthetic (sample of val recordings)")
# ----------------------------------------------------------------------
X_val, y_val, g_val = build_windows_for_recordings(val_ids, labels_by_id)
human_mask = (y_val == 0)
synth_mask = (y_val == 1)
print(f"val windows: {len(y_val)} total, {human_mask.sum()} human, {synth_mask.sum()} synthetic")
d_all = describe(X_val, "ALL val log-mel values")
d_human = describe(X_val[human_mask], "HUMAN val log-mel values")
d_synth = describe(X_val[synth_mask], "SYNTHETIC val log-mel values")
dump["logmel_val_all"] = d_all
dump["logmel_val_human"] = d_human
dump["logmel_val_synth"] = d_synth
dump["n_val_windows"] = {"total": int(len(y_val)), "human": int(human_mask.sum()), "synthetic": int(synth_mask.sum())}


# ----------------------------------------------------------------------
section("5. Log-mel distribution: TRAINING windows vs EXTERNAL (different recording pipeline) windows")
# ----------------------------------------------------------------------
# External set = OpenSLR human + gTTS/espeak synthetic, already processed
# to 8kHz mono by an earlier experiment (external_data/). This is the
# best available PROXY for "audio from a totally different recording
# pipeline/microphone" since a real live mic can't be captured in this
# environment -- explicitly labeled as a proxy, not a live mic capture.
if EXTERNAL_MANIFEST_CSV.exists():
    ext_manifest = pd.read_csv(EXTERNAL_MANIFEST_CSV)
    ext_mels = []
    for _, row in ext_manifest.iterrows():
        wav_path = EXTERNAL_PROCESSED_DIR / f"{row['external_id']}.wav"
        if not wav_path.exists():
            continue
        y_ext, sr_ext = sf.read(str(wav_path), always_2d=True)
        y_ext = y_ext[:, 0]
        windows = subsample_windows(slice_windows(y_ext, sr_ext), MAX_WINDOWS_PER_RECORDING)
        for w in windows:
            ext_mels.append(log_mel_spectrogram(w))
    if ext_mels:
        X_ext = np.stack(ext_mels)
        d_ext = describe(X_ext, "EXTERNAL (proxy for out-of-pipeline audio) log-mel values")
        dump["logmel_external_proxy"] = d_ext
        print("\nCOMPARISON:")
        print(f"  val (in-distribution)      mean={d_all['mean']:.2f} std={d_all['std']:.2f} "
              f"p1={d_all['p1']:.2f} p99={d_all['p99']:.2f}")
        print(f"  external (proxy, OOD)      mean={d_ext['mean']:.2f} std={d_ext['std']:.2f} "
              f"p1={d_ext['p1']:.2f} p99={d_ext['p99']:.2f}")
else:
    print("external_data manifest not found -- skipping (see external_data/README.md)")


# ----------------------------------------------------------------------
section("6. Raw CNN behavior on fixed known windows, NO smoothing/decision/UI")
# ----------------------------------------------------------------------
def raw_scores_for_recording(anon_id, n=6):
    y_audio, sr = load_channel0(anon_id)
    windows = slice_windows(y_audio, sr)
    # take n evenly spaced windows across the recording
    idx = np.linspace(0, len(windows) - 1, min(n, len(windows))).round().astype(int)
    scores = []
    with torch.no_grad():
        for i in idx:
            mel = log_mel_spectrogram(windows[i])
            x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
            prob = torch.sigmoid(model(x)).item()
            scores.append(prob)
    return scores

known_human = [aid for aid in val_ids if labels_by_id[aid] == "human"][:3]
known_synth = [aid for aid in val_ids if labels_by_id[aid] == "synthetic"][:3]
raw_behavior = {}
print("HUMAN recordings (raw CNN scores, no decision layer):")
for aid in known_human:
    scores = raw_scores_for_recording(aid)
    raw_behavior[aid] = {"label": "human", "scores": scores}
    print(f"  {aid}: {[f'{s:.3f}' for s in scores]}  max={max(scores):.3f}")
print("SYNTHETIC recordings (raw CNN scores, no decision layer):")
for aid in known_synth:
    scores = raw_scores_for_recording(aid)
    raw_behavior[aid] = {"label": "synthetic", "scores": scores}
    print(f"  {aid}: {[f'{s:.3f}' for s in scores]}  min={min(scores):.3f}")
dump["raw_cnn_behavior"] = raw_behavior


# ----------------------------------------------------------------------
section("7. Determinism: same tensor through the CNN 10 times")
# ----------------------------------------------------------------------
print(f"model.training = {model.training} (must be False)")
fixed_mel = log_mel_spectrogram(slice_windows(*load_channel0(known_human[0]))[3])
x_fixed = torch.from_numpy(fixed_mel).unsqueeze(0).unsqueeze(0)
runs = []
with torch.no_grad():
    for i in range(10):
        prob = torch.sigmoid(model(x_fixed)).item()
        runs.append(prob)
        print(f"  run {i+1}: {prob:.10f}")
print(f"min={min(runs):.10f} max={max(runs):.10f} std={np.std(runs):.2e}")
dump["determinism_runs"] = runs
dump["model_training_flag"] = model.training


# ----------------------------------------------------------------------
section("8. Most suspicious HUMAN windows (highest raw AI score) + their acoustic features")
# ----------------------------------------------------------------------
suspicious = []
all_human_scores = []
t0 = time.time()
for anon_id in [aid for aid in val_ids if labels_by_id[aid] == "human"]:
    y_audio, sr = load_channel0(anon_id)
    raw_windows = slice_windows(y_audio, sr)  # ALL windows (not subsampled), whole-recording normalized
    with torch.no_grad():
        for i, w in enumerate(raw_windows):
            mel = log_mel_spectrogram(w)
            x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)
            prob = torch.sigmoid(model(x)).item()
            all_human_scores.append(prob)
            suspicious.append((prob, anon_id, i, w))
print(f"scanned {len(all_human_scores)} human windows across {len([a for a in val_ids if labels_by_id[a]=='human'])} val recordings in {time.time()-t0:.0f}s")
describe(all_human_scores, "raw CNN score distribution over ALL human val windows")

suspicious.sort(key=lambda t: -t[0])
top10 = suspicious[:10]
bottom_normal = suspicious[len(suspicious)//2 - 5: len(suspicious)//2 + 5]  # a "typical" middle sample for comparison

def features_for_window(w, sr):
    rms_frames = extract_rms(w)
    centroid, bandwidth, rolloff, zcr = extract_spectral(w, sr)
    _, f0, voiced_flag, _ = extract_f0(w, sr)
    voiced = f0[~np.isnan(f0)]
    return {
        "rms": float(np.mean(rms_frames)),
        "zcr": float(np.mean(zcr)),
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
        "spectral_rolloff": float(np.mean(rolloff)),
        "f0_median": float(np.median(voiced)) if len(voiced) >= 8 else None,
        "n_voiced_frames": int(len(voiced)),
    }

print("\nTOP 10 most-AI-scored HUMAN windows:")
top10_records = []
for prob, anon_id, i, w in top10:
    feats = features_for_window(w, TARGET_SR)
    rec = {"anon_id": anon_id, "window_index": i, "score": prob, **feats}
    top10_records.append(rec)
    print(f"  {anon_id} w{i}: score={prob:.3f} rms={feats['rms']:.4f} zcr={feats['zcr']:.4f} "
          f"centroid={feats['spectral_centroid']:.0f}Hz bandwidth={feats['spectral_bandwidth']:.0f}Hz "
          f"rolloff={feats['spectral_rolloff']:.0f}Hz f0={feats['f0_median']}")

print("\n'TYPICAL' (median-score) HUMAN windows, for comparison:")
typical_records = []
for prob, anon_id, i, w in bottom_normal:
    feats = features_for_window(w, TARGET_SR)
    rec = {"anon_id": anon_id, "window_index": i, "score": prob, **feats}
    typical_records.append(rec)
    print(f"  {anon_id} w{i}: score={prob:.3f} rms={feats['rms']:.4f} zcr={feats['zcr']:.4f} "
          f"centroid={feats['spectral_centroid']:.0f}Hz bandwidth={feats['spectral_bandwidth']:.0f}Hz "
          f"rolloff={feats['spectral_rolloff']:.0f}Hz f0={feats['f0_median']}")

dump["top10_suspicious_human_windows"] = top10_records
dump["typical_human_windows"] = typical_records
dump["all_human_window_scores_summary"] = describe(all_human_scores, "(recorded above)")


# ----------------------------------------------------------------------
section("9. Class balance")
# ----------------------------------------------------------------------
rec_counts = manifest["label"].value_counts().to_dict()
print(f"recording-level: {rec_counts}")
X_train_full, y_train_full, _ = build_windows_for_recordings(train_ids, labels_by_id)
print(f"train window-level: human={int((y_train_full==0).sum())} synthetic={int((y_train_full==1).sum())}")
print(f"val window-level:   human={int(human_mask.sum())} synthetic={int(synth_mask.sum())}")
dump["class_balance"] = {
    "recording_level": rec_counts,
    "train_window_level": {"human": int((y_train_full == 0).sum()), "synthetic": int((y_train_full == 1).sum())},
    "val_window_level": {"human": int(human_mask.sum()), "synthetic": int(synth_mask.sum())},
}

with open(DUMP_PATH, "w") as f:
    json.dump(dump, f, indent=2, default=str)
print(f"\nRaw dump written to {DUMP_PATH}")
