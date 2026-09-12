# Dreamteam-Altur2026

Proyecto de detección de voz sintética / deepfake para llamadas telefónicas. El sistema toma audio estereo de una conversación (llamante y agente), extrae rasgos acústicos, temporales, semánticos y de fase, y usa un modelo de clasificación para decidir si la voz del llamante parece artificial.

El repositorio combina un backend FastAPI, un pipeline de extracción de características y una lógica de entrenamiento del modelo para detectar patrones típicos de audio sintetizado (vocoder, micro-tremor, turn-taking anómalo, trampas semánticas, etc.).

## ¿Qué hace este repo?

La idea principal es detectar si un audio de llamada fue generado o manipulado para simular una voz humana en una operación de seguridad. En la práctica, el proyecto:

- recibe audio base64 o datos de audio en una solicitud HTTP;
- decodifica el WAV/stream a un array NumPy;
- separa los canales: `caller` (canal 0) y `agent` (canal 1);
- extrae más de 40 atributos relacionados con:
  - latencia de turnos y overlap;
  - energía de ruido ambiente;
  - autocorrelación del silencio;
  - MFCC y features espectrales;
  - transcripción semántica con Whisper;
  - micro-tremor y respiración como proxies biológicos;
  - anomalías de fase/vocoder;
- evalúa un modelo entrenado para devolver una probabilidad de synthetic/real;
- dispara un evento simulado de verificación por WhatsApp si la llamada parece sintética.

En resumen, no es un sistema de ASR general; es un detector de fraude de voz orientado a seguridad de llamadas.

## Arquitectura del proyecto

### Estructura principal

```text
Dreamteam-Altur2026/
├── app/
│   ├── __pycache__/
│   ├── audio_utils.py      # extracción de features y decodificación de audio
│   ├── main.py             # API FastAPI principal
│   └── model.py            # pipeline de entrenamiento y otra API de prueba
├── explain_model.py        # utilitario para inspeccionar/explicar el modelo
├── requirements.txt        # dependencias de Python
├── test_api.py             # cliente simple para probar el endpoint /detect
├── README.md               # documentación del proyecto
├── .gitignore              # ignorados del repo
├── audio/                  # esperado para entrenamiento (no siempre incluido)
├── manifest.csv            # metadata de dataset (esperado para entrenamiento)
├── model.pkl               # modelo entrenado (generado en tiempo de ejecución)
├── features_cache.pkl      # cache de features (generado en tiempo de ejecución)
└── .venv/                  # entorno local opcional (no se versiona)
```

### Módulos clave

#### `app/audio_utils.py`

Este archivo contiene la lógica central para:

- decodificar base64 a audio WAV;
- detectar segmentos de voz con VAD (Voice Activity Detection);
- calcular latencias, overlap e indicadores temporales;
- extraer MFCC, ZCR, flatness espectral, centroides, etc.;
- usar `faster-whisper` para transcribir una pista y detectar patrones semánticos sospechosos.

#### `app/model.py`

Aquí se define la lógica de entrenamiento del modelo. El flujo es:

1. leer `manifest.csv`;
2. encontrar cada archivo `.wav`;
3. extraer features con `extract_features`;
4. guardar el resultado en `features_cache.pkl`;
5. entrenar un `HistGradientBoostingClassifier` (o un `XGBClassifier` calibrado en otra variante del proyecto);
6. guardar el modelo en `model.pkl`.

#### `app/main.py`

Es la API principal del proyecto. Expone un servicio FastAPI que recibe audio y devuelve si es sintético o no. Aquí también se simula un webhook de verificación por WhatsApp para casos sospechosos.

#### `explain_model.py`

Utilidad para “explicar” el comportamiento del modelo. No es necesariamente la API principal, sino una herramienta para entender la decisión del sistema sobre una predicción.

## Cómo funciona el entrenamiento

El pipeline de entrenamiento sigue este flujo:

```python
# pseudo-flujo
manifest = pd.read_csv("manifest.csv")
for row in manifest:
    filepath = os.path.join(audio_dir, f"{row['anon_id']}.wav")
    data, sr = sf.read(filepath)
    caller = data[:, 0]
    agent = data[:, 1]
    features = extract_features(caller, agent, sr)
    label = 1 if row["label"] == "synthetic" else 0
    guardar (features, label, row["split"])

X = np.array(features)
y = np.array(labels)
model = HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=42)
model.fit(X[train_mask], y[train_mask])
joblib.dump(model, "model.pkl")
```

### Datos esperados para entrenar

Normalmente el entrenamiento espera una estructura así:

```csv
anon_id,label,split
001,synthetic,train
002,organic,train
003,synthetic,test
...
```

Y además una carpeta `audio/` con archivos `.wav` cuyos nombres coincidan con `anon_id`.

### Cache de features

Si existe `features_cache.pkl`, el sistema intenta reutilizarlo para ahorrar tiempo. Esto hace que re-entrenar sea más rápido, pero solo si el conjunto de datos y los archivos de audio no cambiaron.

### Modelo generado

Cuando no existe `model.pkl`, el sistema lo crea automáticamente al arrancar o al invocar la lógica de entrenamiento. El artefacto final es un modelo serializado que puede cargarse con `joblib`.

## Endpoints de la API

### `GET /`

Devuelve un status básico del servicio.

Ejemplo:

```bash
curl http://127.0.0.1:8000/
```

Respuesta:

```json
{"status": "Altur Voice Deepfake Detector is Active"}
```

### `POST /detect`

Endpoint principal del proyecto. Recibe audio en base64 como `audio_base64` o como `audio`.

Body esperado:

```json
{
  "audio_base64": "<cadena base64 del audio wav>"
}
```

También se acepta:

```json
{
  "audio": "<cadena base64 del audio wav>"
}
```

Validaciones:

- requiere un payload de audio;
- decodifica el audio;
- exige stereo con al menos 2 canales.

Respuesta tipo:

```json
{
  "is_synthetic": true,
  "confidence": 0.87,
  "recommended_action": "trigger_whatsapp_2fa",
  "breakdown": {
    "timing_and_environment": "8.5%",
    "caller_acoustics": "32.4%",
    "agent_acoustics": "0.6%",
    "semantics": "0.0%",
    "biological_and_phase": "58.5%"
  }
}
```

### `POST /predict`

Este endpoint aparece en `app/model.py` y es una variante alternativa del servicio. Recibe dos audios separados:

```json
{
  "caller_audio": "<base64 caller>",
  "agent_audio": "<base64 agent>"
}
```

La respuesta es equivalente, con `is_synthetic`, `confidence`, `recommended_action` y `breakdown`.

## Cómo ejecutar el proyecto

### 1) Crear entorno e instalar dependencias

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Importante: este repo usa varias librerías que no aparecen todas en `requirements.txt` de forma explícita, dependiendo de la ruta que quieras ejecutar:

```bash
pip install faster-whisper xgboost pandas
```

`faster-whisper` es necesario para el análisis semántico de transcripción, y `xgboost`/`pandas` son relevantes si ejecutas la variante de entrenamiento de `app/model.py`.

### 2) Iniciar la API principal

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3) Probar con curl

```bash
curl -X POST "http://127.0.0.1:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{"audio_base64":"<base64>"}'
```

### 4) Probar con el script incluido

```bash
python test_api.py
```

Este script busca el primer archivo `.wav` dentro de `audio/*.wav`, lo convierte a base64 y lo envía a `/detect`.

## Ejemplo de uso en Python

```python
import base64
import requests

with open("audio/example.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "http://127.0.0.1:8000/detect",
    json={"audio_base64": audio_b64}
)

print(response.status_code)
print(response.json())
```

## Qué caracteriza a este detector

El modelo no solo mira la calidad de la voz. Se intenta detectar señales de síntesis combinando varias familias de features:

- Temporales: latencia entre turnos, solapamiento, VAD.
- Ambientales: ruido de fondo, energía de silencio.
- Acústicas: MFCC, zero crossing rate, flatness espectral.
- Semánticas: transcripción con Whisper y métricas de lenguaje artificial (“I understand”, “Let me help”, etc.).
- Biológicas: micro-tremor, respiración, centroides espectrales.
- De fase: artefactos tipo vocoder / patrones matemáticos del sintetizador.

## Observaciones y estado del proyecto

Este repo parece estar en una etapa de prototipo / investigación:

- hay una API funcional de prueba,
- el modelo puede entrenarse a partir de un dataset local,
- la lógica de predicción se basa en features hand-crafted,
- hay código duplicado o variantes experimentales (`app/main.py` y `app/model.py`),
- se simula la integración con WhatsApp para detener una llamada sospechosa.

## TODO / backlog sugerido

Aunque no hay un backlog formal en el repositorio, estas son tareas naturales para llevar el proyecto a un estado más robusto:

- [ ] centralizar una única API de producción (`app/main.py` vs `app/model.py`);
- [ ] documentar el formato exacto de `manifest.csv` y del dataset de audio;
- [ ] añadir validación real de audio (sample rate, canales, duración mínima);
- [ ] guardar logs y métricas de predicciones por llamada;
- [ ] añadir tests unitarios de extracción de features y validación de payloads;
- [ ] reemplazar la simulación de WhatsApp con integración real o webhook configurable;
- [ ] revisar dependencias y dejar `requirements.txt` reproducible y completo;
- [ ] evaluar performance del modelo con métricas reales (precision, recall, F1, ROC-AUC);
- [ ] preparar despliegue con Docker / Docker Compose / CI.

## Recomendación rápida

Si quieres empezar desde cero:

```bash
pip install -r requirements.txt
pip install faster-whisper
uvicorn app.main:app --reload
```

Luego prueba `/detect` con un archivo WAV estereo, preferentemente con audio de llamada con dos canales: llamante y agente.

## Resumen

Dreamteam-Altur2026 es un prototipo de detector de voz deepfake para seguridad de llamadas. Combina audio processing, machine learning y análisis semántico para calificar si una voz es artificial. La pieza central del repo es la extracción de features desde audio estereo, el entrenamiento de un clasificador y la exposición de una API FastAPI para detectar contenido sospechoso en tiempo real.
