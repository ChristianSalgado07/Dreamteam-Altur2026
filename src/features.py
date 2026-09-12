import numpy as np
import librosa
from scipy.signal import medfilt


# ============================================================
# VAD basado en energía
# ============================================================
def get_vad_flags(y, sr, frame_ms=20, energy_threshold=None):
    frame_size = int(sr * frame_ms / 1000)
    hop = frame_size
    energies = []
    for i in range(0, len(y) - frame_size, hop):
        frame = y[i:i + frame_size]
        energies.append(np.sqrt(np.mean(frame ** 2) + 1e-12))
    energies = np.array(energies)
    if energy_threshold is None:
        energy_threshold = max(np.percentile(energies, 30) * 1.5, 1e-4)
    flags = (energies > energy_threshold).astype(bool)
    flags = medfilt(flags.astype(float), kernel_size=5).astype(bool)
    return flags


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
    min_frames = int(60 / frame_ms)
    segments = [(s, e) for s, e in segments if (e - s) >= min_frames]
    return segments


# ============================================================
# Features de comportamiento conversacional
# ============================================================
def extract_behavioral_features(caller, agent, sr):
    frame_ms = 20
    caller_flags = get_vad_flags(caller, sr, frame_ms)
    agent_flags = get_vad_flags(agent, sr, frame_ms)
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

    caller_turn_durs = [(e - s) * frame_ms / 1000.0 for s, e in caller_segs]
    agent_turn_durs = [(e - s) * frame_ms / 1000.0 for s, e in agent_segs]
    avg_caller_turn = np.mean(caller_turn_durs) if caller_turn_durs else 0
    avg_agent_turn = np.mean(agent_turn_durs) if agent_turn_durs else 0
    std_caller_turn = np.std(caller_turn_durs) if caller_turn_durs else 0
    std_agent_turn = np.std(agent_turn_durs) if agent_turn_durs else 0

    latencies = []
    for c_start, c_end in caller_segs:
        prev_agent = [s for s, e in agent_segs if e <= c_start]
        if prev_agent:
            last_agent_end = max(prev_agent)
            latencies.append((c_start - last_agent_end) * frame_ms / 1000.0)
    avg_latency = np.mean(latencies) if latencies else 0
    std_latency = np.std(latencies) if latencies else 0

    interruptions = 0
    for i in range(1, min_len):
        if caller_flags[i] and not caller_flags[i - 1] and agent_flags[i]:
            interruptions += 1

    agent_interruptions = 0
    for i in range(1, min_len):
        if agent_flags[i] and not agent_flags[i - 1] and caller_flags[i]:
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


# ============================================================
# Features acústicas
# ============================================================
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


# ============================================================
# Capa semántica con mlx-whisper (Apple Silicon)
# ============================================================
_whisper_backend = None
_whisper_model = None


def _get_whisper():
    """Detecta y carga el mejor backend de Whisper disponible."""
    global _whisper_backend, _whisper_model
    if _whisper_backend is not None:
        return _whisper_backend, _whisper_model

    try:
        import mlx_whisper  # noqa: F401
        _whisper_backend = "mlx"
        _whisper_model = "mlx-community/whisper-tiny-mlx"
        print("[semantic] Usando mlx-whisper (tiny)")
        return _whisper_backend, _whisper_model
    except ImportError:
        pass

    try:
        import whisper
        _whisper_backend = "openai"
        _whisper_model = whisper.load_model("tiny")
        print("[semantic] Usando openai-whisper (tiny)")
        return _whisper_backend, _whisper_model
    except ImportError:
        raise ImportError(
            "No se encontró Whisper. Instala con: "
            "pip install mlx-whisper  (recomendado en Apple Silicon) "
            "o: pip install openai-whisper"
        )


def _transcribe(path, backend, model):
    if backend == "mlx":
        import mlx_whisper
        result = mlx_whisper.transcribe(
            path,
            path_or_hf_repo=model,
            language="es",
        )
        return result["text"].lower()
    else:
        result = model.transcribe(path, language="es")
        return result["text"].lower()


def extract_semantic_features(caller, agent, sr):
    """Capa semántica: analiza QUÉ dice el llamante comparado con el agente."""
    import tempfile
    import soundfile as sf
    import os

    backend, model = _get_whisper()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, caller, sr)
        caller_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, agent, sr)
        agent_path = f.name

    try:
        caller_text = _transcribe(caller_path, backend, model)
        agent_text = _transcribe(agent_path, backend, model)
    finally:
        try:
            os.unlink(caller_path)
            os.unlink(agent_path)
        except OSError:
            pass

    caller_words = caller_text.split()
    n_caller = max(len(caller_words), 1)
    n_agent = max(len(agent_text.split()), 1)

    # Titubeos / muletillas (humanos sí, TTS no)
    hesitation_words = ["eh", "este", "pues", "bueno", "mmm", "no sé", "es que",
                        "o sea", "digamos", "verdad"]
    hesitation_count = sum(caller_text.count(w) for w in hesitation_words)

    # Repeticiones ("sí sí sí", "no no")
    repetitions = 0
    for i in range(1, len(caller_words)):
        if caller_words[i] == caller_words[i - 1]:
            repetitions += 1

    # Densidad de puntuación (Whisper mete comas y puntos)
    punctuation_count = (caller_text.count(",") + caller_text.count(".")
                         + caller_text.count("..."))

    avg_word_len = np.mean([len(w) for w in caller_words]) if caller_words else 0

    # Palabras "largas" (>10 chars) — LLMs tienden a vocabulario más complejo
    long_words = sum(1 for w in caller_words if len(w) > 10)
    long_word_ratio = long_words / n_caller

    speech_ratio = n_caller / n_agent

    return {
        "caller_word_count": n_caller,
        "agent_word_count": n_agent,
        "caller_avg_word_len": avg_word_len,
        "hesitation_count": hesitation_count,
        "hesitation_ratio": hesitation_count / n_caller,
        "repetitions": repetitions,
        "repetition_ratio": repetitions / n_caller,
        "punct_count": punctuation_count,
        "punct_density": punctuation_count / n_caller,
        "long_word_ratio": long_word_ratio,
        "semantic_speech_ratio": speech_ratio,
        "question_marks_agent": agent_text.count("?") + agent_text.count("¿"),
        "question_marks_caller": caller_text.count("?") + caller_text.count("¿"),
    }


# ============================================================
# Combinadores
# ============================================================
def extract_features_from_arrays(caller, agent, sr, include_semantic=False):
    acoustic = extract_acoustic_features(caller, sr)
    behavioral = extract_behavioral_features(caller, agent, sr)
    features = {**acoustic, **behavioral}
    if include_semantic:
        try:
            semantic = extract_semantic_features(caller, agent, sr)
            features.update(semantic)
        except Exception as e:
            print(f"[WARN] Semantic layer failed: {e}")
    return features


def extract_features(anon_id, include_semantic=False):
    """Extrae features para un solo audio del dataset."""
    from data import load_audio
    caller, agent, sr = load_audio(anon_id)
    return extract_features_from_arrays(caller, agent, sr,
                                        include_semantic=include_semantic)


# ============================================================
# Versión con augmentation (SOLO para entrenamiento)
# ============================================================
def extract_features_with_augmentation(anon_id, n_augments=1, include_semantic=False):
    """
    Devuelve lista: [features_original, features_aug_1, ...]
    
    Las features semanticas se calculan UNA SOLA VEZ desde el audio original
    (el contenido no cambia con augmentation) y se cachean en disco.
    """
    from data import load_audio
    from augment import augment_telephony

    caller, agent, sr = load_audio(anon_id)

    # 1) Extraer TODO (acustico + comportamiento) SIN semantica
    original_no_sem = extract_features_from_arrays(caller, agent, sr, include_semantic=False)

    semantic_feats = None
    if include_semantic:
        # 2) Intentar leer del cache
        try:
            from semantic_cache import get_cached, set_cached
            semantic_feats = get_cached(anon_id)
            if semantic_feats is None:
                # 3) Transcribir con whisper (una sola vez)
                semantic_feats = extract_semantic_features(caller, agent, sr)
                set_cached(anon_id, semantic_feats)
                print(f"  [whisper] {anon_id} procesado y cacheado")
        except Exception as e:
            print(f"  [WARN] semantic failed para {anon_id}: {e}")
            semantic_feats = None

    # 4) Combinar original + semantica
    original = {**original_no_sem}
    if semantic_feats:
        original.update(semantic_feats)
    results = [original]

    # 5) Augmented versions: reutilizan la semantica del original
    for k in range(n_augments):
        aug_caller = augment_telephony(caller, sr, seed=(hash(anon_id) % 100000) + k)
        aug_agent = augment_telephony(agent, sr, seed=(hash(anon_id) % 100000) + 1000 + k)
        aug_feats = extract_features_from_arrays(aug_caller, aug_agent, sr, include_semantic=False)
        if semantic_feats:
            aug_feats.update(semantic_feats)
        results.append(aug_feats)

    return results