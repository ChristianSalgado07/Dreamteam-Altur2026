"""
Simula la llamada que harán los jueces.
Uso:
    python test_api.py <ruta_al_wav>
    python test_api.py # usa un ejemplo por defecto
"""
import base64
import sys
import requests

API_URL = "https://webshots-bomb-lined-doe.trycloudflare.com/detect"
WAV_PATH = "altur-challenge-audio/audio/call_0181ce113ebe.wav"


def probar(ruta):
    with open(ruta, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    resp = requests.post(API_URL, json={"audio_base64": b64})
    print(f"Status: {resp.status_code}")
    print(f"Respuesta: {resp.json()}")


if __name__ == "__main__":
    ruta = sys.argv[1] if len(sys.argv) > 1 else WAV_PATH
    probar(ruta)