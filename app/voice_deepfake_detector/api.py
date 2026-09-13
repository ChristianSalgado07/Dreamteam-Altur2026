"""
API HTTP para detección de voz sintética.
Pipeline V2 (61 features): 39 MFCCs + 10 espectrales + 12 comportamiento.

Servidor web que escucha peticiones HTTP en http://localhost:8000/detect, 
recibe un audio, lo decodifica, corre el modelo y devuelve la predicción.
"""
import base64
import io
import warnings
import numpy as np
import librosa
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import MODEL_PATH, SAMPLE_RATE
from audio_utils import extraer_features_v2

warnings.filterwarnings("ignore")

app = FastAPI(title="Altur Voice Deepfake Detector")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
""" Carga el modelo entrenado. """
modelo = joblib.load(MODEL_PATH)
print(f"[Info] Modelo cargado: {MODEL_PATH}")

# Pesos por categoría del frontend
if hasattr(modelo, "feature_importances_"):
    imp = modelo.feature_importances_
else:
    imp = np.ones(61) / 61.0

W_MFCC = float(imp[0:39].sum())
W_SPECTRAL = float(imp[39:49].sum())
W_BEHAVIOR = float(imp[49:61].sum())
_total = W_MFCC + W_SPECTRAL + W_BEHAVIOR + 1e-12
W_MFCC /= _total
W_SPECTRAL /= _total
W_BEHAVIOR /= _total


class DetectRequest(BaseModel):
    audio_base64: str = None
    audio: str = None


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float
    recommended_action: str
    breakdown: dict

""" base64 -> WAV -> (caller, agente) a 16kHz, normalizados. """
def decodificar_audio_caller(b64_str: str):
    wav_bytes = base64.b64decode(b64_str)
    y, _ = librosa.load(io.BytesIO(wav_bytes), sr=SAMPLE_RATE, mono=False)
    if y.ndim == 1:
        caller, agente = y, np.zeros_like(y)
    else:
        caller = y[0]
        agente = y[1] if y.shape[0] > 1 else np.zeros_like(caller)
    if len(caller) > 0:
        caller = librosa.util.normalize(caller)
    if len(agente) > 0:
        agente = librosa.util.normalize(agente)
    return caller, agente

""" Mapea 3 categorías a las 5 que espera el frontend. """
def construir_breakdown() -> dict:
    return {
        "timing_and_environment": f"{W_BEHAVIOR*100:.1f}%",
        "caller_acoustics":       f"{W_MFCC*100:.1f}%",
        "agent_acoustics":        "0.0%",
        "semantics":              "0.0%",
        "biological_and_phase":   f"{W_SPECTRAL*100:.1f}%",
    }


@app.get("/")
def root():
    return {"status": "Altur Voice Deepfake Detector is Active"}


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    raw_b64 = req.audio_base64 or req.audio
    if not raw_b64:
        raise HTTPException(status_code=400, detail="Missing audio base64 payload")

    try:
        caller, agente = decodificar_audio_caller(raw_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decoding error: {str(e)}")

    if len(caller) == 0:
        raise HTTPException(status_code=400, detail="Empty audio")

    try:
        features = extraer_features_v2(caller, agente).reshape(1, -1)
        pred = int(modelo.predict(features)[0])
        proba = modelo.predict_proba(features)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    is_synthetic = bool(pred == 1)
    confidence = float(max(proba))

    return DetectResponse(
        is_synthetic=is_synthetic,
        confidence=confidence,
        recommended_action="trigger_whatsapp_2fa" if is_synthetic else "proceed_call",
        breakdown=construir_breakdown(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)