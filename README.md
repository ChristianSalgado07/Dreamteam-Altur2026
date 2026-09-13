# Dreamteam · Altur HackMTY 2026

**Detección de voz sintética en llamadas telefónicas bancarias.**

Sistema end-to-end que clasifica si el llamante de una conversación bancaria es un humano real o una voz generada por IA. Entrena un clasificador XGBoost sobre 61 features acústicas y de comportamiento, y expone un endpoint HTTP compatible con el contrato del challenge.

---

## El problema

En Latinoamérica, millones de personas dependen del teléfono como único canal bancario. Con unos segundos de audio público, cualquiera puede clonar una voz lo suficiente para engañar a humanos y máquinas. Los contact centers —humanos o automatizados— no tienen forma robusta de verificar si el que llama es real.

**Nuestro objetivo:** dado un audio de una llamada (canal 0 = caller, canal 1 = agente), decidir si el caller es humano o sintético.

---

## Estructura del repositorio

```
Dreamteam-Altur2026/
│
├── app/
│   └── voice_deepfake_detector/     ← Backend (producción + herramientas)
│       ├── api.py                    (endpoint HTTP)
│       ├── audio_utils.py            (extracción de features)
│       ├── config.py                 (configuración centralizada)
│       ├── train_model.py            (entrena el modelo final)
│       ├── evaluate_model.py         (validación cruzada)
│       ├── check_endpoint.py         (cliente oficial del challenge)
│       ├── test_api.py               (cliente de prueba)
│       ├── Dockerfile                (build de producción)
│       ├── requirements.txt
│       └── outputs/
│           └── modelo_final_audio.pkl
│
├── src/                              ← Frontend React + TypeScript
│   ├── App.tsx
│   ├── components/
│   ├── hooks/
│   ├── lib/
│   └── pages/
│
├── package.json                      ← Configuración del frontend
├── vite.config.ts
├── tailwind.config.js
├── tsconfig.json
├── index.html
├── .env.example
└── README.md
```

### Por qué esta separación

- **Backend y frontend viven separados** porque se despliegan de forma independiente. El backend va a Railway vía Docker. El frontend puede ir a Vercel, Netlify, o servirse estático.
- **El detector vive en su propia carpeta** (`app/voice_deepfake_detector/`) para que su Dockerfile tenga todo lo que necesita sin arrastrar el frontend.
- **El Dockerfile copia solo lo mínimo indispensable** (API, features, configuración y modelo). El resto del proyecto (frontend, dataset, scripts de análisis) no entra al contenedor.
- **`outputs/` guarda el modelo entrenado** porque es un artefacto de build, no código fuente.

---

## Cómo correrlo

### Backend (API)

```bash
cd app/voice_deepfake_detector
pip install -r requirements.txt
python api.py
```

La API queda escuchando en `http://localhost:8000`.

### Frontend (consola visual)

En otra terminal:

```bash
npm install
npm run dev
```

El dashboard queda en `http://localhost:5173`.

### Verificar el endpoint

```bash
# Health check
curl http://localhost:8000/

# Probar un audio del dataset
cd app/voice_deepfake_detector
python test_api.py altur-challenge-audio/audio/call_0181ce113ebe.wav

# Evaluar con el cliente oficial del challenge
python check_endpoint.py \
  --url http://localhost:8000/detect \
  --manifest manifest.csv \
  --audio-dir altur-challenge-audio/audio \
  --split val --n 20
```

### Documentación interactiva

FastAPI genera una UI automática en `http://localhost:8000/docs`.

---

## Deploy

El backend incluye un `Dockerfile` listo para Railway, Render, o cualquier PaaS que soporte Docker.

### Por qué Docker

Desplegar sin Docker implica que el servidor tenga las versiones correctas de Python, `ffmpeg`, `libsndfile`, y todas las dependencias. Docker garantiza que el contenedor arranca **exactamente igual** en cualquier servidor. Los jueces corren lo mismo que nosotros.

### Por qué las capas importan

El Dockerfile está ordenado para que el **build sea rápido y la imagen pequeña**. Estos dos objetivos se logran separando el contenido en capas que Docker cachea de forma independiente:

- **Las dependencias del sistema y de Python cambian poco.** Al aislarlas en sus propias capas, Docker las reutiliza entre builds. Esto convierte un redeploy de 5-10 minutos en uno de 30 segundos.
- **El código de la aplicación cambia seguido.** Al copiarlo en una capa al final, solo esa capa se reconstruye cuando hay cambios. Las dependencias quedan cacheadas.
- **Solo entra lo indispensable.** El contenedor no necesita el frontend, el dataset, ni los scripts de análisis. Copiar únicamente los 6 archivos que la API usa mantiene la imagen ligera (~500 MB vs ~1.5 GB) y el arranque rápido.

### Configuración en Railway

- Root Directory: `app/voice_deepfake_detector`
- Dockerfile Path: `./Dockerfile`
- Port: dinámico vía `$PORT` (Railway lo inyecta automáticamente)

### Build y run local

```bash
cd app/voice_deepfake_detector
docker build -t altur-detector .
docker run -p 8000:8000 altur-detector
```

---

## Nuestro enfoque

Combinamos **tres capas de features ortogonales**:

### 1. Capa acústica (39 features)

39 MFCCs del canal del caller, extraídos con librosa: media de los 13 coeficientes (timbre promedio), desviación estándar (variabilidad temporal), y delta MFCCs (velocidad de cambio entre fonemas).

**Insight:** el timbre promedio solo no discrimina. La variabilidad y la dinámica son más informativas.

### 2. Capa espectral (10 features)

Cinco propiedades espectrales de segundo orden, cada una con media y std: Zero-Crossing Rate, Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, y RMS Energy.

**Insight:** los TTS modernos imitan bien los MFCCs promedio, pero las features espectrales de segundo orden son más difíciles de falsificar.

### 3. Capa de comportamiento conversacional (12 features)

Usamos un VAD por energía adaptativo para segmentar los turnos de cada canal, y extraemos: ratios de habla, overlap, número de turnos, duración media y variabilidad, latencias de respuesta, e interrupciones de cada lado.

**Insight:** los humanos interrumpimos, dudamos, reaccionamos con timing irregular. Los sintéticos siguen patrones demasiado limpios. La inconsistencia es la firma de la humanidad.

---

## Resultados

### Validación cruzada estratificada (5-fold)

| Métrica | Valor |
|---|---|
| Balanced accuracy | 98.51% ± 1.90% |
| ROC AUC | 0.9990 ± 0.0013 |
| Brier score | 0.0109 |

### Hold-out honesto (train → val, speaker-disjoint)

Entrenando solo con el split `train` y evaluando sobre `val` (voces que el modelo nunca vio):

| Métrica | Valor |
|---|---|
| Balanced accuracy | 100% (71/71) |
| ROC AUC | 1.000 |
| Brier score | 0.000 |

### Cliente oficial del challenge (`check_endpoint.py`)

Evaluado vía HTTP real contra el endpoint público:

| Métrica | Valor |
|---|---|
| Llamadas procesadas | 71 |
| Errores | 0 |
| Latencia media | 2.97 s |
| Latencia máxima | 6.10 s |

**Nota sobre las métricas:** el 100% en el hold-out honesto tiene un intervalo de confianza del 95% de aproximadamente [95%, 100%] dado el tamaño de 71 muestras. La CV de 5 folds (98.51%) es la estimación más conservadora.

---

## Endpoint HTTP

Contrato compatible con el challenge:

```
POST /detect
Content-Type: application/json

{
  "audio_base64": "<WAV estéreo 8kHz codificado en base64>",
  "call_id": "call_xxx",
  "sample_rate": 8000,
  "channels": 2
}
```

Respuesta (HTTP 200):

```json
{
  "is_synthetic": true,
  "confidence": 0.998,
  "recommended_action": "trigger_whatsapp_2fa",
  "breakdown": {
    "timing_and_environment": "10.9%",
    "caller_acoustics": "31.4%",
    "agent_acoustics": "0.0%",
    "semantics": "0.0%",
    "biological_and_phase": "57.7%"
  }
}
```

Campos:

| Campo | Tipo | Descripción |
|---|---|---|
| is_synthetic | bool | true si el caller es IA, false si es humano |
| confidence | float | Certeza del veredicto (0 a 1) |
| recommended_action | string | trigger_whatsapp_2fa o proceed_call |
| breakdown | dict | Importancia relativa de cada categoría |

---

## Decisiones de diseño

### ¿Por qué XGBoost y no un modelo end-to-end?

Con 353 audios de entrenamiento, un modelo end-to-end (tipo Wav2Vec2) sobreajustaría. XGBoost sobre features interpretables es más robusto en datasets pequeños y nos permite explicar las decisiones.

### ¿Por qué features de comportamiento conversacional?

Los clasificadores acústicos pueden ser engañados por TTS modernos. Pero un TTS no puede simular la dinámica conversacional de un humano sin acceso al canal del agente en tiempo real. Es una señal ortogonal a la acústica.

### ¿Por qué XGBoost y no Random Forest?

Probamos ambos. XGBoost ganó por 1 punto porcentual y es más rápido en inferencia. La diferencia no es dramática, pero es consistente.

### ¿Por qué confidence = max(proba)?

El challenge define confidence como "la certeza del veredicto", no la probabilidad de sintético. Usar `max(proba)` asegura que la confianza refleja la certeza correctamente tanto en humanos como en sintéticos.

---

## Equipo

**Dreamteam** · HackMTY 2026

- Christian Salgado
- Ángel Luna
- Tamara Padilla
- Michelle Lagos

**Última actualización:** Septiembre 2026