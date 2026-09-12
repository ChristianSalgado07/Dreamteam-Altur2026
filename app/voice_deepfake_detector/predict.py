"""
Predice si un audio es humano o sintético.
Uso:
    python predict.py <anon_id>
    python predict.py              # usa un ejemplo del manifest
"""
import sys
import joblib
import pandas as pd
from config import MODEL_PATH, MANIFEST_PATH
from audio_utils import cargar_audio, extraer_features


def predecir_audio(anon_id, modelo):
    # Usa el mismo pipeline que el entrenamiento: canal 0 + normalización
    audio = cargar_audio(anon_id, canal=0, normalizar=True)
    features = extraer_features(audio).reshape(1, -1)
    pred = modelo.predict(features)[0]
    proba = modelo.predict_proba(features)[0]
    etiqueta = "SYNTHETIC (IA)" if pred == 1 else "HUMAN"

    print(f"\nanon_id: {anon_id}")
    print(f"Predicción: {etiqueta}")
    print(f"Confianza: {max(proba)*100:.2f}%")
    print(f"  - Human:     {proba[0]*100:.2f}%")
    print(f"  - Synthetic: {proba[1]*100:.2f}%")
    return etiqueta


def main():
    modelo = joblib.load(MODEL_PATH)

    if len(sys.argv) > 1:
        predecir_audio(sys.argv[1], modelo)
    else:
        df = pd.read_csv(MANIFEST_PATH)
        ejemplo = df.iloc[0]
        print(f"[INFO] Etiqueta real: {ejemplo['label']}")
        predecir_audio(ejemplo["anon_id"], modelo)


if __name__ == "__main__":
    main()