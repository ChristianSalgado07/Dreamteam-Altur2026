from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class DetectionPayload(BaseModel):
    audio_base64: str = None

@app.get("/")
def read_root():
    return {"status": "Altur Challenge API is running"}

@app.post("/detect")
def detect(payload: DetectionPayload):
    # TODO: Connect feature extraction and ML model here
    return {
        "is_synthetic": True, 
        "confidence": 0.85
    }