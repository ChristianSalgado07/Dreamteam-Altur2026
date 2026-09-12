from fastapi import FastAPI, Request, HTTPException
import base64
import io
import soundfile as sf
import numpy as np
import pandas as pd
from predict import Detector
from features import extract_features_from_arrays

app = FastAPI()
detector = Detector()

@app.get("/")
def root():
    return {"status": "ok", "service": "deepfake-voice-detector"}

@app.post("/detect")
async def detect(request: Request):
    # 1) Aceptar JSON {"audio": "..."} o body raw base64
    audio_b64 = None
    try:
        data = await request.json()
        audio_b64 = data.get("audio") or data.get("wav") or data.get("file")
    except Exception:
        pass

    if audio_b64 is None:
        body = await request.body()
        audio_b64 = body.decode('utf-8').strip()

    if not audio_b64:
        raise HTTPException(status_code=400, detail="No audio provided")

    # 2) Decodificar base64
    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")

    # 3) Leer WAV
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes), always_2d=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid WAV")

    # 4) Re-muestrear a 8kHz si es necesario
    if sr != 8000:
        import librosa
        y = librosa.resample(y.T, orig_sr=sr, target_sr=8000).T
        sr = 8000

    # 5) Separar canales
    caller = y[:, 0]
    agent = y[:, 1] if y.shape[1] > 1 else np.zeros_like(caller)

    # 6) Predecir
    try:
        is_synthetic, prob = detector.predict_from_arrays(caller, agent, sr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return {"is_synthetic": is_synthetic, "confidence": float(prob)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)