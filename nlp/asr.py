"""
Local/offline ASR wrapper (faster-whisper). No audio is sent to any
external/cloud API — the model runs entirely on this machine after its
weights are downloaded once from the public faster-whisper/HuggingFace
model repo (a one-time model download, not per-recording transcription
traffic).

Language is forced to Spanish ("es") since the dataset's filler-word
vocabulary (eh, este, pues, o sea, ...) and manual spot-checks confirm
the calls are in Spanish — this avoids occasional language
misdetection on short/noisy turns.
"""
import numpy as np
from scipy.signal import resample_poly

from preprocessing.audio_io import EXPECTED_SAMPLE_RATE

MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"
LANGUAGE = "es"
TARGET_SR = 16000  # what faster-whisper/Whisper expects for a raw ndarray input

_model = None


def get_model():
    """Lazily loaded, one instance per process (see transcribe_pipeline.py's
    multiprocessing worker initializer — loading the model is the expensive
    part, ~40s, so each worker process does it once and reuses it)."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
    return _model


def resample_to_16k(y: np.ndarray, orig_sr: int = EXPECTED_SAMPLE_RATE) -> np.ndarray:
    if orig_sr == TARGET_SR:
        return y.astype(np.float32)
    return resample_poly(y, TARGET_SR, orig_sr).astype(np.float32)


def transcribe_array(y_16k: np.ndarray) -> dict:
    """
    Transcribes one already-16kHz-mono-float32 audio array. Returns text
    plus ASR provenance/quality fields: avg_logprob (higher/less negative
    = more confident) and no_speech_prob (higher = more likely this
    segment is not actually speech — useful for flagging hallucination
    or channel-contamination candidates later).
    """
    model = get_model()
    if len(y_16k) < int(0.05 * TARGET_SR):  # shorter than 50ms: not worth sending to the model
        return {
            "text": "", "n_segments": 0, "avg_logprob": np.nan,
            "no_speech_prob": np.nan, "language": None, "language_probability": np.nan,
        }

    segments, info = model.transcribe(
        y_16k, language=LANGUAGE, beam_size=1, vad_filter=False, condition_on_previous_text=False,
    )
    segments = list(segments)
    if not segments:
        return {
            "text": "", "n_segments": 0, "avg_logprob": np.nan,
            "no_speech_prob": np.nan, "language": info.language,
            "language_probability": info.language_probability,
        }

    text = " ".join(s.text.strip() for s in segments).strip()
    return {
        "text": text,
        "n_segments": len(segments),
        "avg_logprob": float(np.mean([s.avg_logprob for s in segments])),
        "no_speech_prob": float(np.mean([s.no_speech_prob for s in segments])),
        "language": info.language,
        "language_probability": float(info.language_probability),
    }
