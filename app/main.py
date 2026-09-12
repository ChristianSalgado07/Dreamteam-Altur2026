from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.audio_utils import decode_base64_audio, extract_features, compute_breakdown
from app.model import load_or_train_model, load_feature_stats

app = FastAPI(title="Altur Voice Deepfake Detection API")

# CORS — allows the frontend (localhost:5173) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and feature stats at startup
model = load_or_train_model()
feature_stats = load_feature_stats()


class DetectionPayload(BaseModel):
    audio_base64: str = None
    audio: str = None


def trigger_whatsapp_2fa(confidence: float):
    """Simulates hitting Altur's WhatsApp API for identity verification."""
    payload = {
        "bot_action": "halt_call",
        "user_notification": f"SECURITY ALERT: We detected synthetic voice patterns ({confidence*100:.1f}%). Please reply VERIFY to confirm your identity.",
        "status": "dispatched"
    }
    print(f"\n🚨 [ALTUR WEBHOOK SIMULATION] 🚨\nPayload dispatched: {payload}\n")


@app.get("/")
def read_root():
    return {"status": "Altur Voice Deepfake Detector is Active"}


@app.post("/detect")
def detect(payload: DetectionPayload, background_tasks: BackgroundTasks):
    raw_b64 = payload.audio_base64 or payload.audio
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

    # Extract the 40-feature vector
    features = extract_features(caller, agent, sr)

    # Predict probability
    prob_synthetic = float(model.predict_proba([features])[0][1])
    is_synthetic = bool(prob_synthetic >= 0.50)

    # Dynamic breakdown (per-prediction, not hardcoded)
    if feature_stats is not None:
        breakdown = compute_breakdown(
            features,
            feature_stats["importances"],
            feature_stats["means"],
            feature_stats["stds"],
        )
    else:
        breakdown = {
            "timing_and_environment": "0.0%",
            "caller_acoustics": "0.0%",
            "agent_acoustics": "0.0%",
            "semantics": "0.0%",
            "biological_and_phase": "0.0%",
        }

    # Fire the WhatsApp webhook silently in the background if deepfake
    if is_synthetic:
        background_tasks.add_task(trigger_whatsapp_2fa, prob_synthetic)

    return {
        "is_synthetic": is_synthetic,
        "confidence": prob_synthetic,
        "recommended_action": "trigger_whatsapp_2fa" if is_synthetic else "proceed_call",
        "breakdown": breakdown,
    }