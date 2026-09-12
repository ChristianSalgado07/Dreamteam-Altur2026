import io
import base64
import numpy as np
import soundfile as sf
import librosa

def decode_base64_audio(b64_string: str):
    """Decodes base64 string to a stereo NumPy array and sample rate."""
    audio_bytes = base64.b64decode(b64_string)
    data, sr = sf.read(io.BytesIO(audio_bytes))
    return data, sr

def extract_features(caller: np.ndarray, agent: np.ndarray, sr: int = 8000) -> np.ndarray:
    """Extracts MFCCs to quickly map the acoustic shape of both voices."""
    # n_mfcc=13 is the standard baseline for human voice analysis
    mfcc_caller = librosa.feature.mfcc(y=caller, sr=sr, n_mfcc=13)
    mfcc_agent = librosa.feature.mfcc(y=agent, sr=sr, n_mfcc=13)
    
    # Average the coefficients across time to create a flat, 1D feature vector
    caller_mean = np.mean(mfcc_caller, axis=1)
    agent_mean = np.mean(mfcc_agent, axis=1)
    
    # Combine both channels into a single 26-feature array for the classifier
    return np.concatenate((caller_mean, agent_mean))