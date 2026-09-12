# HackMTY26 — Defensa contra Voice Deepfakes

**Clasificador en tiempo real de voz humana vs. sintética en llamadas telefónicas.** Detecta si el llamante es real o una clonación con precisión 96.88%, combinando acústica, comportamiento conversacional y semántica.

---

## El problema

La voz dejó de ser prueba de identidad. Con unos segundos de audio público, cualquiera puede clonar una voz lo suficiente para engañar a humanos y máquinas.  
En Latinoamérica, millones de personas dependen del teléfono como único canal bancario — y son exactamente las más expuestas a fraude de identidad y suplantación. Los contact centers, humanos o con IA, no tienen forma robusta de verificar si el que llama es real.

---

## Nuestra solución: tres capas ortogonales

Nuestro enfoque no se apoya en una sola señal (que puede ser falsificada), sino en **tres fuentes de evidencia ortogonales** que capturan aspectos diferentes de la voz humana.

### 1. Capa acústica: "Los humanos son inconsistentes"

Extraemos **40 coeficientes MFCC + delta**, **spectral bandwidth**, **rolloff**, **zero-crossing rate (ZCR)** y **RMS energy**.  
Para cada uno calculamos media y desviación estándar.

**Insight**: La voz humana tiene **alta variabilidad intra-frase**. Los motores TTS (text-to-speech) son demasiado consistentes — ponen síntesis en el acorde, y eso se ve en los estadísticos.

| Feature | Humano | Sintético | Razón |
|---------|--------|-----------|-------|
| `zcr_std` | 0.101 | 0.044 | **2.3×** |
| `spec_bw_std` | 366 Hz | 192 Hz | **1.9×** |
| `spec_rolloff_std` | 1052 Hz | 669 Hz | **1.6×** |

### 2. Capa de comportamiento conversacional: "Los robots respetan turnos"

Aprovechamos que **ambos canales de la llamada** están disponibles: el del llamante (canal 0) y el del agente (canal 1).  
Medimos:

- **Duración de turnos** y promedio por cada uno
- **Latencias de respuesta** del llamante tras cada turno del agente
- **Solapamiento de voz** (habla simultánea)
- **Interrupciones**: cuando el llamante empieza a hablar mientras el agente sigue

**Insight**: Un sistema automático está programado para esperar. Un humano es caótico — interrumpe, se solapa, reacciona con latencias variable y errática.

| Feature | Humano | Sintético | Razón |
|---------|--------|-----------|-------|
| `caller_interruptions` | 28.7 | 15.2 | **1.9×** |
| `avg_caller_turn` (s) | 0.99 | 1.85 | **0.5×** |
| `response_latency_std` (ms) | 347 | 112 | **3.1×** |

### 3. Capa semántica: "Las IA no dudan"

Usamos **mlx-whisper** (optimizado para Apple Silicon) para transcribir ambos canales en paralelo.  
Extraemos:

- **Titubeos** ("eh", "um", "uh", "mmm") — frecuencia y ratio
- **Repeticiones** de palabras ("la la la cosa") y frases
- **Velocidad de locución** (palabras por minuto)
- **Duración de pausas** entre frases

**Insight**: Los humanos dudan, tantean, repiten. Las IA generan frases "limpias" — Whisper lo ve.

| Feature | Humano | Sintético | Razón |
|---------|--------|-----------|-------|
| `repetition_ratio` | 0.289 | 0.070 | **4.1×** ⭐ |
| `hesitation_count` | 14.2 | 2.3 | **6.2×** |
| `speech_rate` (wps) | 158 | 172 | 0.91× |

---

## 📊 Métricas finales (5-fold cross-validation)

| Métrica | Valor |
|---------|-------|
| **ROC AUC** | 0.9974 ± 0.0020 |
| **Accuracy** | 0.9688 ± 0.0115 |
| **Brier score** | 0.0147 |
| **Latencia (p95)** | 2.1s |

El modelo usa **Random Forest calibrado (500 árboles, calibración sigmoid)** entrenado en **~500 llamadas** (humanas + sintéticas) con **~80 features** ortogonales.

---

## Decisiones de diseño

### Augmentation telefónica
Entrenar solo con "datos limpios" es frágil. Simulamos **condiciones reales del teléfono**:
- **Filtro bandpass**: 300–3400 Hz (ancho de banda telefónico)
- **Ruido blanco**: SNR 15–30 dB (50% del tiempo)
- **Ganancia aleatoria**: ±3 dB

Esto dobla el dataset efectivo sin necesidad de nuevas llamadas.

### Calibración + Temperature softening
El modelo crudo da probabilidades muy extremas (0.95 / 0.05). Aplicamos:

1. **Calibración sigmoid** en train (reduce Brier score ~40%)
2. **Temperature scaling** en inferencia (T=3):
   $$p_{\text{soft}} = \frac{1}{1 + e^{-\text{logit}(p_{\text{raw}}) / T}}$$

   Esto da confianzas "útiles" para desempates (e.g., si `soft` ∈ [0.45, 0.55], pedir verificación adicional).

### Cache de Whisper en disco
Transcribir cada llamada (2 minutos) toma ~2 horas en GPU. **Cacheamos en pickle** por `anon_id` — segunda ejecución es 30 segundos.

---

## Cómo correr

### Instalación

```bash
cd /Users/christian/workspace/Dreamteam-Altur2026

# Crear entorno virtual (opcional)
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# (Opcional) Para Whisper con Apple Silicon:
# pip install mlx-whisper
```

### Entrenamiento

```bash
cd src
python train.py
```

Genera `models/model.pkl` y `models/feature_names.pkl`.  
**Parámetros ajustables** en `train.py`:
- `INCLUDE_SEMANTIC`: incluir features de Whisper (más lento, mejor AUC)
- `USE_AUGMENTATION`: duplicar dataset con augmentation telefónica
- `N_AUGMENTS`: número de augments por llamada (default 1)

### Validación

```bash
python eval_val.py
```

Evalúa en split `val`. Muestra ROC AUC, Brier score, distribuciones crudo vs. suavizado.

### Diagnóstico de features

```bash
python diagnose.py
```

Muestra importancias de cada feature según el modelo entrenado.

### Endpoint FastAPI

```bash
python app.py
```

Servidor en `http://localhost:8000`.

**Request**:
```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d "{\"audio\": \"$(base64 < call.wav)\"}"
```

**Response**:
```json
{
  "is_synthetic": false,
  "confidence": 0.892
}
```

### Test del endpoint

```bash
python test_endpoint.py 10          # 10 muestras del split 'val'
python test_endpoint.py 5 --train   # 5 muestras del split 'train'
```

---

## 📁 Estructura del proyecto

```
Dreamteam-Altur2026/
├── README.md                  # Este archivo
├── requirements.txt           # Dependencias
│
├── data/
│   ├── manifest.csv          # Metadatos: anon_id, label, split
│   ├── raw/                  # Archivos WAV de entrada
│   └── semantic_cache/       # Cache pickle de features Whisper
│
├── models/
│   ├── model.pkl             # Modelo Random Forest calibrado
│   └── feature_names.pkl     # Nombres de features (para reindex)
│
└── src/
    ├── app.py                # FastAPI endpoint
    ├── train.py              # Pipeline de entrenamiento
    ├── crossval.py           # Cross-validation manual (5-fold)
    ├── eval_val.py           # Evaluación en split 'val'
    ├── diagnose.py           # Feature importance + estadísticas
    ├── predict.py            # Clase Detector + temperature softening
    ├── features.py           # Extracción de 80 features (acústica + comportamiento + semántica)
    ├── data.py               # Carga de WAVs y manifest
    ├── augment.py            # Augmentation telefónica
    ├── semantic_cache.py     # Gestión de cache de Whisper
    └── test_endpoint.py      # Cliente para probar endpoint
```

---

## API Endpoint: `/detect`

**POST** `http://localhost:8000/detect`

### Request

```json
{
  "audio": "SUQzBAAAI1ITPQAcAAAAI0lORk8AAAAPAAAABgAA... (base64 de WAV estéreo)"
}
```

### Response (éxito)

```json
{
  "is_synthetic": false,
  "confidence": 0.9124
}
```

| Campo | Tipo | Rango | Significado |
|-------|------|-------|------------|
| `is_synthetic` | bool | — | `true` = voz sintética, `false` = voz humana |
| `confidence` | float | [0.05, 0.95] | Confianza posterior (tras temperature softening) |

### Comportamiento de entrada

- Acepta **JSON** (`{"audio": "..."}`) o **raw body** base64
- Espera WAV **estéreo** (canal 0 = llamante, canal 1 = agente)
- Remuestrea automáticamente a **8 kHz** si es necesario
- Normaliza amplitud antes de procesar

### Códigos de error

| Status | Detail |
|--------|--------|
| 400 | "No audio provided" — JSON vacío o body faltante |
| 400 | "Invalid base64" — Decodificación falló |
| 400 | "Invalid WAV" — `soundfile` no pudo leerlo |
| 500 | "Inference error: ..." — Fallo en extracción de features |

---

## Créditos

**Equipo Dreamteam**  
Christian Salgado
Ángel Luna
Tamara Padilla
Michelle Lagos

---