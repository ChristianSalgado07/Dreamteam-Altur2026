import io
import base64
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import correlate
from faster_whisper import WhisperModel
import re

# 'tiny.en' is extremely fast and runs cleanly on a CPU
whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

def extract_semantic_features(audio_array: np.ndarray, sr: int = 8000) -> list:
    """Transcribes audio and extracts cognitive/semantic traps."""
    # Transcribe the caller's track
    segments, _ = whisper_model.transcribe(audio_array, beam_size=1)
    transcript = " ".join([segment.text for segment in segments]).lower()

    # 1. The Helpful Assistant Trap
    ai_tells = ["i apologize", "i understand", "how can i assist", "let me help", "i can certainly", "sure"]
    ai_tells_count = float(sum(transcript.count(tell) for tell in ai_tells))

    # 2. The Articulation Trap (Lack of human fillers)
    human_fillers = [" um ", " uh ", " like ", " you know ", " i mean "]
    filler_count = float(sum(transcript.count(filler) for filler in human_fillers))

    # 3. Verbosity / Hallucination Trap
    word_count = float(len(transcript.split()))
    
    return [ai_tells_count, filler_count, word_count]

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
    """Extracts Temporal, Environmental, Acoustic, Semantic, Biological, and Phase features."""
    caller_intervals, caller_vad, caller_rms = compute_vad_intervals(caller, sr)
    agent_intervals, agent_vad, agent_rms = compute_vad_intervals(agent, sr)

    # 1. Turn-Transition Latencies
    latencies = [c_start - a_end for a_start, a_end in agent_intervals for c_start, _ in caller_intervals if c_start >= a_end]
    mean_ttl = float(np.mean(latencies)) if latencies else 0.5
    std_ttl = float(np.std(latencies)) if latencies else 0.0
    
    # 2. Barge-in / Overlap Dynamics
    min_len = min(len(caller_vad), len(agent_vad))
    overlap_frames = np.sum((caller_vad[:min_len]) & (agent_vad[:min_len]))
    overlap_ratio = float(overlap_frames / (np.sum((caller_vad[:min_len]) | (agent_vad[:min_len])) + 1e-6))

    # 3. Ambient Noise Autocorrelation
    silent_indices = np.where(~caller_vad[:len(caller_rms)])[0]
    if len(silent_indices) > 50:
        silence_rms = caller_rms[silent_indices]
        silence_mean_energy, silence_std_energy = float(np.mean(silence_rms)), float(np.std(silence_rms))
        norm_silence = silence_rms - silence_mean_energy
        autocorr = correlate(norm_silence, norm_silence, mode='full')[len(norm_silence)-1:]
        peak_autocorr = float(np.max(autocorr[1:] / (autocorr[0] + 1e-6))) if len(autocorr) > 1 else 0.0
    else:
        silence_mean_energy, silence_std_energy, peak_autocorr = 0.0, 0.0, 0.0

    # 4. Acoustic Features (MFCCs)
    caller_mfcc_mean = np.mean(librosa.feature.mfcc(y=caller, sr=sr, n_mfcc=13), axis=1)
    agent_mfcc_mean = np.mean(librosa.feature.mfcc(y=agent, sr=sr, n_mfcc=13), axis=1)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(caller)))
    spec_flatness = float(np.mean(librosa.feature.spectral_flatness(y=caller)))

    # 5. Semantic Traps
    semantic_metrics = extract_semantic_features(caller, sr)

    # 6. Biological Incongruence (Stress Tremors & Breathing)
    # Humans have vocal micro-tremors (centroid volatility) and inhale during VAD silences
    centroid = librosa.feature.spectral_centroid(y=caller, sr=sr)[0]
    micro_tremor_variance = float(np.std(centroid))
    breathing_proxy = float(silence_mean_energy) if silence_mean_energy > 0.0001 else 0.0 

    # 7. Phase Anomalies (Vocoder Footprint)
    # Synthetic vocoders leave mathematical patterns in phase derivatives 
    stft_caller = librosa.stft(caller)
    phase_diff = np.diff(np.angle(stft_caller), axis=1)
    phase_volatility = float(np.var(phase_diff))

    return np.concatenate((
        [mean_ttl, std_ttl, overlap_ratio, silence_mean_energy, silence_std_energy, peak_autocorr, zcr, spec_flatness],
        caller_mfcc_mean,
        agent_mfcc_mean,
        semantic_metrics,
        [micro_tremor_variance, breathing_proxy, phase_volatility]
    )).astype(np.float32)