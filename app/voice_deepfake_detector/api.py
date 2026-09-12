"""
API HTTP para detección de voz sintética.

Arrancar:
    python api.py

Probar (en otra terminal):
    python test_api.py <ruta_al_wav>
"""
import base64
import io
import warnings
import numpy as np
import librosa
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from config import MODEL_PATH, SAMPLE_RATE
from audio_utils import extraer_features

warnings.filterwarnings("ignore")

app = FastAPI(title="Altur Voice Deepfake Detector")

modelo = joblib.load(MODEL_PATH)
print(f"[INFO] Modelo cargado: {MODEL_PATH}")


class DetectRequest(BaseModel):
    audio_base64: str


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float
    prob_human: float
    prob_synthetic: float


def decodificar_audio_caller(b64_str: str) -> np.ndarray:
    """base64 -> WAV -> canal 0 a 16kHz, normalizado."""
    wav_bytes = base64.b64decode(b64_str)
    y, _ = librosa.load(io.BytesIO(wav_bytes), sr=SAMPLE_RATE, mono=False)
    if y.ndim == 2:
        y = y[0]
    if len(y) > 0:
        y = librosa.util.normalize(y)
    return y


@app.get("/")
def root():
    return {"status": "ok", "model": "xgboost-audio-v1"}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    try:
        caller_audio = decodificar_audio_caller(req.audio_base64)
        if len(caller_audio) == 0:
            raise ValueError("Audio vacío")
        features = extraer_features(caller_audio).reshape(1, -1)
        pred = int(modelo.predict(features)[0])
        proba = modelo.predict_proba(features)[0]
        return DetectResponse(
            is_synthetic=bool(pred == 1),
            confidence=float(max(proba)),
            prob_human=float(proba[0]),
            prob_synthetic=float(proba[1]),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)