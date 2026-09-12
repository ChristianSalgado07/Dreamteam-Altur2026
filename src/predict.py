import joblib
import numpy as np
import pandas as pd
from features import extract_features_from_arrays
from config import MODEL_DIR
import os

# Rango final permitido
CONF_MIN = 0.05
CONF_MAX = 0.95

# Temperatura para suavizar la confianza en espacio logit.
# T=1.0 → sin suavizado (probs crudas)
# T=2.0 → suavizado moderado
# T=3.0 → suavizado fuerte (recomendado si todo sale 0.95/0.05)
TEMPERATURE = 3.0


def _soften(p, T=TEMPERATURE):
    """Suaviza probabilidad p en espacio logit con temperatura T."""
    p = np.clip(p, 1e-7, 1 - 1e-7)
    logit = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-logit / T))


class Detector:
    def __init__(self, model_path=None, feature_names_path=None):
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "model.pkl")
        if feature_names_path is None:
            feature_names_path = os.path.join(MODEL_DIR, "feature_names.pkl")

        self.model = joblib.load(model_path)

        if os.path.exists(feature_names_path):
            self.feature_names = joblib.load(feature_names_path)
        else:
            self.feature_names = None

        self.use_semantic = (
            self.feature_names is not None
            and any(
                fn.startswith("hesitation") or fn.startswith("caller_word")
                for fn in self.feature_names
            )
        )

    def predict_from_arrays(self, caller, agent, sr):
        feats = extract_features_from_arrays(
            caller, agent, sr, include_semantic=self.use_semantic
        )
        X = pd.DataFrame([feats])

        if self.feature_names is not None:
            X = X.reindex(columns=self.feature_names, fill_value=0)
        X = X.fillna(0)

        raw_prob = float(self.model.predict_proba(X)[0, 1])

        # 1) Suavizar en logit
        soft = _soften(raw_prob)

        # 2) Clip defensivo por si acaso
        soft = max(CONF_MIN, min(CONF_MAX, soft))

        return bool(soft > 0.5), soft
