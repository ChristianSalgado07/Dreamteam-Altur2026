import joblib
import numpy as np
import pandas as pd
from features import extract_features_from_arrays
from config import MODEL_DIR
import os

class Detector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "model.pkl")
        self.model = joblib.load(model_path)

    def predict_from_arrays(self, caller, agent, sr):
        feats = extract_features_from_arrays(caller, agent, sr)
        X = pd.DataFrame([feats]).fillna(0)
        prob = self.model.predict_proba(X)[0, 1]
        return bool(prob > 0.5), float(prob)