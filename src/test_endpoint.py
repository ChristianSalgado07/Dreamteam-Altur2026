"""
Prueba el endpoint /detect con N llamadas del dataset.
Uso:
    python test_endpoint.py              # prueba 5 de val
    python test_endpoint.py 10           # prueba 10
    python test_endpoint.py 5 --train    # prueba del train split
"""
import sys
import os
import base64
import requests
import pandas as pd
from config import RAW_DIR, MANIFEST_PATH

URL = "http://localhost:8000/detect"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    split = 'train' if '--train' in sys.argv else 'val'

    df = pd.read_csv(MANIFEST_PATH)
    subset = df[df['split'] == split].sample(n=min(n, len(df[df['split'] == split])), random_state=42)

    correct = 0
    results = []

    for _, row in subset.iterrows():
        path = os.path.join(RAW_DIR, f"{row['anon_id']}.wav")
        if not os.path.exists(path):
            print(f"[SKIP] {path} no existe")
            continue

        with open(path, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode('utf-8')

        try:
            r = requests.post(URL, json={"audio": audio_b64}, timeout=120)
            resp = r.json()
        except Exception as e:
            print(f"[ERR] {row['anon_id']}: {e}")
            continue

        pred = resp['is_synthetic']
        true = (row['label'] == 'synthetic')
        ok = (pred == true)
        correct += int(ok)

        results.append({
            'id': row['anon_id'],
            'true': row['label'],
            'pred': 'synthetic' if pred else 'human',
            'conf': resp.get('confidence', None),
            'ok': ok,
        })

        mark = "✅" if ok else "❌"
        conf_str = f"{resp.get('confidence', 0):.3f}" if resp.get('confidence') else "n/a"
        print(f"{mark} {row['anon_id']} | real={row['label']:<10} pred={'synthetic' if pred else 'human':<10} conf={conf_str}")

    print(f"\n=== RESUMEN ===")
    print(f"Aciertos: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")


if __name__ == "__main__":
    main()