import io
import base64
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import correlate

def decode_base64_audio(b64_string: str):
    """Decodes base64 string to a stereo NumPy array and sample rate."""
    audio_bytes = base64.b64decode(b64_string)
    data, sr = sf.read(io.BytesIO(audio_bytes))
    return data, sr

def compute_vad_intervals(audio_mono: np.ndarray, sr: int = 8000, frame_len: int = 256, hop_len: int = 128, threshold: float = 0.015):
    """Energy-based Voice Activity Detection returning speech segments and RMS curve."""
    rms = librosa.feature.rms(y=audio_mono, frame_length=frame_len, hop_length=hop_len)[0]
    is_speech = rms > threshold
    
    intervals = []
    in_speech = False
    start_frame = 0
    for i, active in enumerate(is_speech):
        if active and not in_speech:
            in_speech = True
            start_frame = i
        elif not active and in_speech:
            in_speech = False
            intervals.append((start_frame * hop_len / sr, i * hop_len / sr))
    if in_speech:
        intervals.append((start_frame * hop_len / sr, len(audio_mono) / sr))
    return intervals, is_speech, rms

def extract_features(caller: np.ndarray, agent: np.ndarray, sr: int = 8000) -> np.ndarray:
    """Extracts Pillar 1 (Dynamics) and Pillar 2 (Acoustic) features from stereo audio."""
    caller_intervals, caller_vad, caller_rms = compute_vad_intervals(caller, sr)
    agent_intervals, agent_vad, agent_rms = compute_vad_intervals(agent, sr)

    # 1. Turn-Transition Latencies (Agent stops -> Caller speaks)
    latencies = []
    for a_start, a_end in agent_intervals:
        subsequent = [c_start - a_end for c_start, _ in caller_intervals if c_start >= a_end]
        if subsequent:
            latencies.append(min(subsequent))
    
    mean_ttl = float(np.mean(latencies)) if latencies else 0.5
    std_ttl = float(np.std(latencies)) if latencies else 0.0
    
    # 2. Barge-in / Overlap Dynamics
    min_len = min(len(caller_vad), len(agent_vad))
    overlap_frames = np.sum((caller_vad[:min_len]) & (agent_vad[:min_len]))
    total_speech_frames = np.sum((caller_vad[:min_len]) | (agent_vad[:min_len])) + 1e-6
    overlap_ratio = float(overlap_frames / total_speech_frames)

    # 3. Silence / Ambient Noise Autocorrelation (Detect looping background noise)
    silent_indices = np.where(~caller_vad[:len(caller_rms)])[0]
    if len(silent_indices) > 50:
        silence_rms = caller_rms[silent_indices]
        silence_mean_energy = float(np.mean(silence_rms))
        silence_std_energy = float(np.std(silence_rms))
        
        norm_silence = silence_rms - np.mean(silence_rms)
        autocorr = correlate(norm_silence, norm_silence, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        peak_autocorr = float(np.max(autocorr[1:] / (autocorr[0] + 1e-6))) if len(autocorr) > 1 else 0.0
    else:
        silence_mean_energy = 0.0
        silence_std_energy = 0.0
        peak_autocorr = 0.0

    # 4. Micro-prosody & Pitch stability (PYIN on Caller)
    try:
        f0, voiced_flag, _ = librosa.pyin(caller, fmin=60, fmax=400, sr=sr, frame_length=512)
        f0_clean = f0[voiced_flag & ~np.isnan(f0)]
        f0_mean = float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0
        f0_std = float(np.std(f0_clean)) if len(f0_clean) > 0 else 0.0
        f0_diff = np.abs(np.diff(f0_clean))
        jitter = float(np.mean(f0_diff)) if len(f0_diff) > 0 else 0.0
    except Exception:
        f0_mean, f0_std, jitter = 0.0, 0.0, 0.0

    # 5. Spectral Flatness & Zero Crossing Rate
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(caller)))
    spec_flatness = float(np.mean(librosa.feature.spectral_flatness(y=caller)))

    return np.array([
        mean_ttl, std_ttl, overlap_ratio,
        silence_mean_energy, silence_std_energy, peak_autocorr,
        f0_mean, f0_std, jitter, zcr, spec_flatness
    ], dtype=np.float32)