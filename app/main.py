from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.audio_utils import decode_base64_audio, extract_features
from app.model import load_or_train_model

app = FastAPI()

# Load calibrated model at startup
model = load_or_train_model()

class DetectionPayload(BaseModel):
    audio_base64: str = None
    audio: str = None

@app.get("/")
def read_root():
    return {"status": "Altur Voice Deepfake Detector is Active"}

@app.post("/detect")
def detect(payload: dict):
    raw_b64 = payload.get("audio_base64") or payload.get("audio")
    if not raw_b64:
        raise HTTPException(status_code=400, detail="Missing audio base64 payload")
    
    try:
        data, sr = decode_base64_audio(raw_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio decoding error: {str(e)}")
        
    if len(data.shape) < 2 or data.shape[1] < 2:
        raise HTTPException(status_code=400, detail="Expected 2-channel stereo audio (Ch 0: Caller, Ch 1: Agent)")

    caller = data[:, 0]
    agent = data[:, 1]

    # Extract interaction and acoustic features
    features = extract_features(caller, agent, sr)

    # Predict calibrated probability
    prob_synthetic = float(model.predict_proba([features])[0][1])
    is_synthetic = bool(prob_synthetic >= 0.50)
    confidence = float(prob_synthetic if is_synthetic else (1.0 - prob_synthetic))

    return {
        "is_synthetic": is_synthetic,
        "confidence": round(confidence, 4)
    }