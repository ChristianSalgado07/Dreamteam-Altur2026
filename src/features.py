import numpy as np
import librosa
import webrtcvad
from scipy.signal import medfilt

def float_to_pcm16(y):
    y = np.clip(y, -1, 1)
    return (y * 32767).astype(np.int16)

def get_vad_flags(y, sr, aggressiveness=2, frame_ms=20):
    vad = webrtcvad.Vad(aggressiveness)
    pcm = float_to_pcm16(y)
    frame_size = int(sr * frame_ms / 1000)
    flags = []
    for i in range(0, len(pcm) - frame_size, frame_size):
        frame = pcm[i:i+frame_size]
        flags.append(vad.is_speech(frame.tobytes(), sr))
    return np.array(flags)

def vad_to_segments(flags, frame_ms=20):
    segments = []
    in_speech = False
    start = 0
    for i, f in enumerate(flags):
        if f and not in_speech:
            start = i
            in_speech = True
        elif not f and in_speech:
            segments.append((start, i))
            in_speech = False
    if in_speech:
        segments.append((start, len(flags)))
    return segments

def extract_behavioral_features(caller, agent, sr):
    frame_ms = 20
    caller_flags = get_vad_flags(caller, sr)
    agent_flags = get_vad_flags(agent, sr)
    min_len = min(len(caller_flags), len(agent_flags))
    caller_flags = caller_flags[:min_len]
    agent_flags = agent_flags[:min_len]

    caller_segs = vad_to_segments(caller_flags, frame_ms)
    agent_segs = vad_to_segments(agent_flags, frame_ms)

    total_time = min_len * frame_ms / 1000.0
    caller_speech = np.sum(caller_flags) * frame_ms / 1000.0
    agent_speech = np.sum(agent_flags) * frame_ms / 1000.0
    overlap = np.sum(caller_flags & agent_flags) * frame_ms / 1000.0

    caller_turns = len(caller_segs)
    agent_turns = len(agent_segs)

    caller_turn_durs = [(e-s)*frame_ms/1000.0 for s,e in caller_segs]
    agent_turn_durs = [(e-s)*frame_ms/1000.0 for s,e in agent_segs]
    avg_caller_turn = np.mean(caller_turn_durs) if caller_turn_durs else 0
    avg_agent_turn = np.mean(agent_turn_durs) if agent_turn_durs else 0
    std_caller_turn = np.std(caller_turn_durs) if caller_turn_durs else 0
    std_agent_turn = np.std(agent_turn_durs) if agent_turn_durs else 0

    latencies = []
    for c_start, c_end in caller_segs:
        prev_agent = [s for s,e in agent_segs if e <= c_start]
        if prev_agent:
            last_agent_end = max(prev_agent)
            latencies.append((c_start - last_agent_end) * frame_ms / 1000.0)
    avg_latency = np.mean(latencies) if latencies else 0
    std_latency = np.std(latencies) if latencies else 0

    interruptions = 0
    for i in range(1, min_len):
        if caller_flags[i] and not caller_flags[i-1] and agent_flags[i]:
            interruptions += 1

    agent_interruptions = 0
    for i in range(1, min_len):
        if agent_flags[i] and not agent_flags[i-1] and caller_flags[i]:
            agent_interruptions += 1

    return {
        "total_time": total_time,
        "caller_speech_ratio": caller_speech / total_time if total_time > 0 else 0,
        "agent_speech_ratio": agent_speech / total_time if total_time > 0 else 0,
        "overlap_ratio": overlap / total_time if total_time > 0 else 0,
        "caller_turns": caller_turns,
        "agent_turns": agent_turns,
        "avg_caller_turn": avg_caller_turn,
        "avg_agent_turn": avg_agent_turn,
        "std_caller_turn": std_caller_turn,
        "std_agent_turn": std_agent_turn,
        "avg_latency": avg_latency,
        "std_latency": std_latency,
        "caller_interruptions": interruptions,
        "agent_interruptions": agent_interruptions,
        "caller_turn_rate": caller_turns / total_time if total_time > 0 else 0,
        "agent_turn_rate": agent_turns / total_time if total_time > 0 else 0,
    }

def extract_acoustic_features(caller, sr):
    mfcc = librosa.feature.mfcc(y=caller, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)
    delta_std = np.std(delta, axis=1)
    spec_cent = librosa.feature.spectral_centroid(y=caller, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=caller, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=caller, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(caller)
    rms = librosa.feature.rms(y=caller)

    features = {}
    for i, v in enumerate(mfcc_mean):
        features[f"mfcc_mean_{i}"] = v
    for i, v in enumerate(mfcc_std):
        features[f"mfcc_std_{i}"] = v
    for i, v in enumerate(delta_mean):
        features[f"delta_mean_{i}"] = v
    for i, v in enumerate(delta_std):
        features[f"delta_std_{i}"] = v
    features["spec_cent_mean"] = np.mean(spec_cent)
    features["spec_cent_std"] = np.std(spec_cent)
    features["spec_bw_mean"] = np.mean(spec_bw)
    features["spec_bw_std"] = np.std(spec_bw)
    features["spec_rolloff_mean"] = np.mean(spec_rolloff)
    features["spec_rolloff_std"] = np.std(spec_rolloff)
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)
    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)
    return features

def extract_features_from_arrays(caller, agent, sr):
    acoustic = extract_acoustic_features(caller, sr)
    behavioral = extract_behavioral_features(caller, agent, sr)
    return {**acoustic, **behavioral}

def extract_features(anon_id):
    from data import load_audio
    caller, agent, sr = load_audio(anon_id)
    return extract_features_from_arrays(caller, agent, sr)