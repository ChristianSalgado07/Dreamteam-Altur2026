from fastapi import FastAPI, Request, HTTPException
import base64
import io
import soundfile as sf
import numpy as np
import pandas as pd
from predict import Detector
from features import extract_acoustic_features, extract_behavioral_features

app = FastAPI()
detector = Detector()

@app.post("/detect")
async def detect(request: Request):
    try:
        data = await request.json()
        audio_b64 = data.get("audio") or data.get("wav") or data.get("file")
        if audio_b64 is None:
            body = await request.body()
            audio_b64 = body.decode('utf-8')
    except:
        body = await request.body()
        audio_b64 = body.decode('utf-8')

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")

    try:
        y, sr = sf.read(io.BytesIO(audio_bytes), always_2d=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid WAV")

    if sr != 8000:
        import librosa
        y = librosa.resample(y.T, orig_sr=sr, target_sr=8000).T
        sr = 8000

    caller = y[:, 0]
    agent = y[:, 1] if y.shape[1] > 1 else np.zeros_like(caller)

    feats = extract_features_from_arrays(caller, agent, sr)
    X = pd.DataFrame([feats]).fillna(0)
    prob = detector.model.predict_proba(X)[0, 1]
    is_synthetic = bool(prob > 0.5)

    return {"is_synthetic": is_synthetic, "confidence": float(prob)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)