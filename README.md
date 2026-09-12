# Altur Voice Deepfake Detection Console

Sistema experimental de detección de voces sintéticas para llamadas telefónicas bancarias. El proyecto combina:

- un backend **FastAPI** que recibe un WAV estéreo codificado en Base64;
- extracción de **40 características acústicas, temporales, semánticas y de fase**;
- un clasificador `HistGradientBoostingClassifier` entrenado con audio etiquetado;
- un análisis dinámico por categorías basado en z-scores e importancia por permutación;
- una consola web **React + Vite + TypeScript** para cargar llamadas y visualizar el veredicto;
- una simulación de acción de seguridad por WhatsApp cuando se detecta una voz sintética.

> **Estado actual:** prototipo funcional para demostración/hackathon. No debe usarse como único mecanismo de autenticación o bloqueo de llamadas en producción sin validación con datos representativos, métricas de seguridad, controles de privacidad y revisión humana.

---

## Tabla de contenidos

1. [Qué hace el sistema](#qué-hace-el-sistema)
2. [Arquitectura](#arquitectura)
3. [Estructura del repositorio](#estructura-del-repositorio)
4. [Requisitos](#requisitos)
5. [Instalación](#instalación)
6. [Configuración](#configuración)
7. [Cómo ejecutar](#cómo-ejecutar)
8. [Flujo completo de una detección](#flujo-completo-de-una-detección)
9. [Extracción de características](#extracción-de-características)
10. [Entrenamiento del modelo](#entrenamiento-del-modelo)
11. [API principal: `app.main`](#api-principal-appmain)
12. [API experimental: `app.model`](#api-experimental-appmodel)
13. [Uso del frontend](#uso-del-frontend)
14. [Pruebas y validación](#pruebas-y-validación)
15. [Artefactos y archivos ignorados](#artefactos-y-archivos-ignorados)
16. [Limitaciones conocidas](#limitaciones-conocidas)
17. [Siguientes pasos](#siguientes-pasos)

---

## Qué hace el sistema

Altur analiza una grabación de una llamada bancaria y estima si la voz del **llamante** presenta patrones compatibles con síntesis o manipulación. La grabación debe ser estéreo:

| Canal | Significado |
| --- | --- |
| Canal 0 | Voz del llamante |
| Canal 1 | Voz del agente |

El backend devuelve:

- `is_synthetic`: veredicto booleano;
- `confidence`: probabilidad estimada de la clase sintética, entre `0` y `1`;
- `recommended_action`: continuar la llamada o iniciar una verificación por WhatsApp;
- `breakdown`: contribución relativa de cinco categorías del análisis.

El frontend presenta estos datos como una consola de operaciones de seguridad: gauge radial, barras de señales, radar comparativo, recomendación de acción e historial local.

### Importante sobre las “seis categorías”

El backend implementa **cinco categorías reales** que agrupan el vector de 40 features:

1. Timing & Environment
2. Caller Acoustics
3. Agent Acoustics
4. Semantic Analysis
5. Biological & Phase

La interfaz agrega una sexta dimensión visual llamada **Verdict Confidence**, derivada de `confidence`. No es una categoría adicional calculada por el modelo ni aparece en el JSON original del backend.

---

## Arquitectura

```text
Archivo WAV estéreo
        │
        ▼
Frontend React
  - valida WAV y canales
  - convierte a Base64
  - POST /detect
        │
        ▼
FastAPI (app.main)
  - decodifica audio
  - separa caller/agent
  - calcula 40 features
  - obtiene probabilidad del modelo
  - calcula breakdown dinámico
  - simula webhook de WhatsApp
        │
        ▼
Respuesta JSON
  - veredicto
  - confianza
  - breakdown
  - acción recomendada
```

### Componentes principales

- **Procesamiento de audio:** `soundfile`, `librosa`, `scipy`.
- **Transcripción semántica:** `faster-whisper` con el modelo `tiny`, ejecución CPU/int8.
- **Machine learning:** `scikit-learn` y `HistGradientBoostingClassifier`.
- **Persistencia del modelo:** archivos serializados con `joblib`.
- **API:** FastAPI + Uvicorn.
- **Frontend:** React, Vite, TypeScript, Tailwind CSS, Recharts, Framer Motion, Axios, React Dropzone y Lucide.

---

## Estructura del repositorio

```text
Dreamteam-Altur2026/
├── app/
│   ├── audio_utils.py       # Decodificación y extracción de las 40 features
│   ├── main.py              # API recomendada: GET / y POST /detect
│   └── model.py             # Entrenamiento y API experimental POST /predict
├── src/
│   ├── components/
│   │   ├── ui.tsx           # Primitivas Card, Button y Badge
│   │   ├── Header.tsx
│   │   ├── UploadZone.tsx
│   │   ├── ResultPanel.tsx
│   │   ├── ConfidenceGauge.tsx
│   │   ├── BreakdownBars.tsx
│   │   ├── RadarAnalysis.tsx
│   │   ├── ActionBanner.tsx
│   │   ├── LoadingAnalysis.tsx
│   │   └── HistoryTable.tsx
│   ├── hooks/
│   │   ├── useBackendStatus.ts
│   │   └── useAnalysisHistory.ts
│   ├── lib/
│   │   ├── api.ts            # Cliente Axios
│   │   ├── types.ts          # Tipos de respuesta e historial
│   │   └── utils.ts          # Utilidades y metadata visual
│   ├── pages/
│   │   └── Dashboard.tsx    # Página principal de la SPA
│   ├── App.tsx
│   ├── main.tsx
│   ├── index.css
│   └── vite-env.d.ts
├── explain_model.py         # Variante experimental de API/modelo calibrado
├── test_api.py              # Cliente Python simple para /detect
├── requirements.txt         # Dependencias Python base
├── package.json             # Dependencias y scripts del frontend
├── package-lock.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── tsconfig.json
├── tsconfig.app.json
├── index.html
├── .env.example             # VITE_API_URL
└── .gitignore
```

Los siguientes archivos pueden existir localmente, pero están ignorados por Git:

```text
audio/              # WAV del dataset o de pruebas
manifest.csv        # manifiesto de entrenamiento
features_cache.pkl  # features extraídas
model.pkl           # modelo entrenado
feature_stats.pkl   # medias, desviaciones e importancias
data/
turns/
.venv/
.hf_cache/
node_modules/
dist/
```

---

## Requisitos

### Backend

- Python 3.10+ recomendado.
- Un entorno virtual (`venv`, `conda` o equivalente).
- Audio WAV legible por `soundfile`.
- CPU suficiente para ejecutar extracción acústica y Whisper.
- Conexión a internet en el primer uso de `faster-whisper` para descargar el modelo `tiny`, salvo que ya exista en la caché local.

El archivo `requirements.txt` contiene las dependencias base del backend, pero el código actualmente importa además `pandas`, `joblib` y `faster-whisper`. Para una instalación funcional del backend instala también esos paquetes:

```bash
pip install -r requirements.txt
pip install pandas joblib faster-whisper
```

`xgboost` solo es necesario para ejecutar la variante experimental descrita en [API experimental](#api-experimental-appmodel):

```bash
pip install xgboost
```

El script `test_api.py` utiliza `requests`, que tampoco está declarado en `requirements.txt`:

```bash
pip install requests
```

### Frontend

- Node.js 18+ recomendado.
- npm.
- Backend disponible en `http://localhost:8000` o en la URL configurada mediante `VITE_API_URL`.

---

## Instalación

### 1. Clonar y entrar al repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd Dreamteam-Altur2026
```

### 2. Crear y activar el entorno Python

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar el backend

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pandas joblib faster-whisper requests
```

Para usar `app.model` o `explain_model.py`:

```bash
pip install xgboost
```

### 4. Instalar el frontend

```bash
npm install
```

### 5. Configurar la URL del backend

```bash
cp .env.example .env
```

El `.env` generado contiene:

```env
VITE_API_URL=http://localhost:8000
```

En Windows se puede crear el archivo manualmente con ese contenido. Las variables `VITE_*` se incorporan durante el build/desarrollo de Vite; reinicia el servidor después de cambiar `.env`.

---

## Cómo ejecutar

Se recomienda abrir dos terminales desde la raíz del proyecto.

### Terminal 1: backend recomendado

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

En Windows PowerShell, activa primero `.venv` con el comando correspondiente y ejecuta el mismo comando de Uvicorn.

El servidor estará disponible en:

- API: <http://localhost:8000>
- Swagger UI: <http://localhost:8000/docs>
- OpenAPI JSON: <http://localhost:8000/openapi.json>

### Terminal 2: frontend en desarrollo

```bash
npm run dev
```

La consola normalmente estará en <http://localhost:5173>.

### Build de producción del frontend

```bash
npm run build
```

Esto ejecuta `tsc -b` y después `vite build`. El resultado se genera en `dist/`.

Para previsualizar el build:

```bash
npm run preview
```

---

## Flujo completo de una detección

1. El usuario arrastra o selecciona un archivo `.wav` en `UploadZone`.
2. El navegador restringe la selección a WAV y valida que el audio tenga al menos dos canales mediante `AudioContext`.
3. El frontend lee el archivo con `FileReader` y elimina el prefijo `data:audio/...;base64,`.
4. Axios envía `{ "audio_base64": "..." }` a `POST /detect`.
5. FastAPI decodifica el Base64 con `base64` y carga el audio con `soundfile`.
6. `app.main` verifica que existan dos canales.
7. El canal 0 se procesa como `caller` y el canal 1 como `agent`.
8. `extract_features` genera el vector `float32` de 40 posiciones.
9. El modelo calcula `predict_proba`; la clase sintética se activa cuando la probabilidad es `>= 0.50`.
10. `compute_breakdown` compara las features con las estadísticas del entrenamiento y genera porcentajes por categoría.
11. Si el veredicto es sintético, se agenda `trigger_whatsapp_2fa` como tarea de fondo. Actualmente solo imprime un payload simulado en la terminal.
12. El frontend muestra el gauge, las barras, el radar y la acción recomendada.
13. El resultado se guarda en `localStorage` para formar el historial de la consola.

---

## Extracción de características

`app/audio_utils.py` define el orden exacto del vector:

| Índices | Categoría | Cantidad | Contenido |
| --- | --- | ---: | --- |
| `0:8` | `timing_and_environment` | 8 | Latencias, solapamiento, energía y ruido |
| `8:21` | `caller_acoustics` | 13 | Media de MFCC del llamante |
| `21:34` | `agent_acoustics` | 13 | Media de MFCC del agente |
| `34:37` | `semantics` | 3 | Indicadores derivados de transcripción |
| `37:40` | `biological_and_phase` | 3 | Micro-tremor, respiración y fase |

Total: **40 features**.

### Timing & Environment

- latencia media entre el final del turno del agente y el inicio del turno del llamante;
- desviación estándar de esas latencias;
- proporción de frames con solapamiento entre caller y agent;
- energía media del silencio;
- desviación estándar de la energía del silencio;
- pico de autocorrelación del ruido durante silencios;
- zero-crossing rate;
- spectral flatness.

La actividad de voz se estima mediante un VAD basado en energía RMS, con `frame_len=256`, `hop_len=128` y umbral `0.015`.

### Caller/Agent Acoustics

Se calculan 13 coeficientes MFCC promedio para cada canal. Esto permite comparar el perfil espectral de la voz del llamante y del agente sin mezclar ambas pistas.

### Semantic Analysis

El llamante se transcribe con:

```python
WhisperModel("tiny", device="cpu", compute_type="int8")
```

Después se calculan:

- conteo de expresiones consideradas “AI tells”;
- conteo de muletillas humanas;
- número total de palabras transcritas.

La lista actual de frases está en `extract_semantic_features`. El modelo Whisper es multilingüe, aunque las frases configuradas actualmente están mayormente en inglés.

### Biological & Phase

- variación del centroide espectral como proxy de micro-tremor;
- energía de silencio como proxy de respiración;
- varianza de diferencias de fase del STFT como huella potencial de vocoder.

---

## Entrenamiento del modelo

El entrenamiento está implementado en `app/model.py` mediante `train_and_save_model`.

### Dataset esperado

La estructura esperada es:

```text
audio/
├── call_001.wav
├── call_002.wav
└── ...
manifest.csv
```

El manifiesto debe incluir al menos:

```csv
anon_id,label,split
call_001,synthetic,train
call_002,organic,train
call_003,synthetic,test
```

Para cada fila, el código busca:

```text
audio/{anon_id}.wav
```

El valor `label` se convierte así:

- `synthetic` → clase `1`;
- cualquier otro valor → clase `0` (por convención, `organic`).

La columna `split` se usa para entrenar únicamente con las filas cuyo valor sea `train`.

### Pasos del entrenamiento

1. Si existe `features_cache.pkl`, se carga ese cache.
2. Si no existe, se lee `manifest.csv` y se procesan los WAV en paralelo con `joblib`.
3. Los archivos inexistentes o corruptos se omiten.
4. Se construyen `X`, `y` y la máscara de entrenamiento.
5. Se entrena:

   ```python
   HistGradientBoostingClassifier(
       max_iter=100,
       max_depth=5,
       random_state=42,
   )
   ```

6. Se calculan medias y desviaciones estándar de las features de entrenamiento.
7. Se calcula `permutation_importance` con 5 repeticiones.
8. Se guardan el modelo en `model.pkl` y las estadísticas en `feature_stats.pkl`.

### Entrenamiento manual

Desde la raíz del repositorio:

```bash
python -c "from app.model import train_and_save_model; train_and_save_model()"
```

También se pueden especificar rutas:

```bash
python -c "from app.model import train_and_save_model; train_and_save_model('ruta/manifest.csv', 'ruta/audio')"
```

### Carga automática al iniciar

`app.main` ejecuta al importar:

```python
model = load_or_train_model()
feature_stats = load_feature_stats()
```

Si existen `model.pkl` y `feature_stats.pkl`, se cargan. Si falta alguno, el backend intenta entrenar y puede requerir `manifest.csv`, `audio/` y `features_cache.pkl`.

El primer arranque puede ser lento porque:

- puede extraer features del dataset;
- puede calcular importancias por permutación;
- `faster-whisper` puede descargar el modelo `tiny`.

### Reentrenamiento después de cambiar el dataset

El cache no se invalida automáticamente. Si cambia el dataset o el código de features, elimina manualmente los artefactos generados y vuelve a entrenar:

```bash
rm -f features_cache.pkl model.pkl feature_stats.pkl
python -c "from app.model import train_and_save_model; train_and_save_model()"
```

En Windows PowerShell:

```powershell
Remove-Item features_cache.pkl, model.pkl, feature_stats.pkl -ErrorAction SilentlyContinue
python -c "from app.model import train_and_save_model; train_and_save_model()"
```

---

## API principal: `app.main`

La API recomendada se ejecuta con:

```bash
uvicorn app.main:app --reload --port 8000
```

### CORS

La API habilita CORS para todos los orígenes:

```python
allow_origins=["*"]
```

Esto facilita el desarrollo local con Vite, pero debe restringirse a dominios conocidos antes de un despliegue real.

### `GET /`

Health check utilizado por el frontend cada 10 segundos.

Request:

```bash
curl http://localhost:8000/
```

Response:

```json
{
  "status": "Altur Voice Deepfake Detector is Active"
}
```

### `POST /detect`

Endpoint principal. Acepta cualquiera de estos cuerpos:

```json
{
  "audio_base64": "<BASE64_DEL_WAV_ESTEREO>"
}
```

o:

```json
{
  "audio": "<BASE64_DEL_WAV_ESTEREO>"
}
```

La respuesta normal tiene este formato:

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

#### Acciones posibles

| Acción | Condición |
| --- | --- |
| `trigger_whatsapp_2fa` | `confidence >= 0.50` |
| `proceed_call` | `confidence < 0.50` |

#### Errores esperados

`400 Bad Request`:

```json
{ "detail": "Missing audio base64 payload" }
```

```json
{ "detail": "Audio decoding error: ..." }
```

```json
{ "detail": "Expected 2-channel stereo audio (Ch 0: Caller, Ch 1: Agent)" }
```

Un error durante la extracción o la predicción puede producir un error `500` del servidor.

#### Ejemplo con `curl`

Genera una cadena Base64 y envíala:

```bash
BASE64_AUDIO=$(base64 < audio/example.wav | tr -d '\n')

curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d "{\"audio_base64\":\"${BASE64_AUDIO}\"}"
```

En macOS, el comando `base64` anterior es compatible con archivos pequeños. Para automatizar sin problemas de quoting, también puedes usar `test_api.py`.

---

## API experimental: `app.model`

`app/model.py` también define una aplicación FastAPI separada con `POST /predict`. No es la ruta que consume el frontend actual y no incluye el breakdown dinámico de `app.main`.

Para levantarla:

```bash
uvicorn app.model:app --reload --port 8001
```

Request:

```json
{
  "caller_audio": "<BASE64_DEL_AUDIO_DEL_CALLER>",
  "agent_audio": "<BASE64_DEL_AUDIO_DEL_AGENT>"
}
```

La respuesta mantiene las claves principales:

```json
{
  "is_synthetic": false,
  "confidence": 0.21,
  "recommended_action": "proceed_call",
  "breakdown": {
    "timing_and_environment": "8.5%",
    "caller_acoustics": "32.4%",
    "agent_acoustics": "0.6%",
    "semantics": "0.0%",
    "biological_and_phase": "58.5%"
  }
}
```

Esta implementación usa `XGBClassifier` calibrado con `CalibratedClassifierCV` cuando entrena desde `features_cache.pkl`, mientras que la API recomendada usa el `HistGradientBoostingClassifier` de `app.model.train_and_save_model`. Son variantes experimentales distintas; evita mezclarlas o sobrescribir `model.pkl` sin controlar qué implementación lo generó.

`explain_model.py` contiene esencialmente esta variante experimental y requiere `xgboost`. Actualmente no existe una herramienta de explicación separada conectada a la interfaz principal.

---

## Uso del frontend

### Arranque

```bash
npm install
npm run dev
```

La URL del backend se toma de:

```env
VITE_API_URL=http://localhost:8000
```

Si no está definida, el cliente Axios usa `http://localhost:8000`.

### Funcionalidades

- header con marca Altur y estado del backend;
- health check automático cada 10 segundos;
- dropzone para WAV;
- validación de archivo y número de canales en el navegador;
- nombre, tamaño y duración del audio seleccionado;
- botón `Analyze Call`;
- estado de carga con animación;
- gauge semicircular con aguja animada;
- barras animadas para el breakdown;
- radar chart con baseline humano visual;
- banner de acción para verificación o continuación de llamada;
- panel de detalles técnicos;
- historial de hasta 20 análisis en `localStorage`;
- carga de un resultado previo al hacer click en el historial;
- layout responsive para desktop, tablet y móvil;
- focus rings y labels accesibles en elementos principales.

### Persistencia del historial

El frontend usa la clave:

```text
altur-analysis-history
```

El historial solo guarda resultados y metadata local (`filename`, timestamp, veredicto y confianza); no guarda el archivo de audio.

---

## Pruebas y validación

### Cliente Python incluido

`test_api.py` busca el primer WAV en `audio/*.wav`, lo codifica y llama a `/detect`:

```bash
python test_api.py
```

Requiere:

- backend ejecutándose en `http://127.0.0.1:8000`;
- carpeta `audio/` con al menos un `.wav`;
- paquete `requests`.

### Build del frontend

```bash
npm run build
```

El build realiza:

1. comprobación TypeScript con `tsc -b`;
2. bundle de producción con Vite.

No hay actualmente una suite de tests automatizados de Python o React configurada en el repositorio.

### Verificaciones manuales recomendadas

1. `GET /` responde correctamente.
2. Un WAV mono devuelve `400`.
3. Un WAV estéreo válido llega a `/detect`.
4. La respuesta contiene las cinco categorías del breakdown.
5. La UI cambia a `Backend connected`.
6. Una respuesta sintética muestra la alerta roja y agenda el log del webhook simulado.
7. Un resultado se conserva después de recargar el navegador.

---

## Artefactos y archivos ignorados

`.gitignore` excluye, entre otros:

- grabaciones (`*.wav`);
- manifests CSV y datos de entrenamiento (`*.csv`);
- artefactos de ML (`*.pkl`);
- entornos Python (`.venv/`);
- caché de Hugging Face (`.hf_cache/`);
- dependencias y build del frontend (`node_modules/`, `dist/`);
- variables locales (`.env`, `.env.local`);
- metadata de TypeScript (`*.tsbuildinfo`).

Por eso una instalación nueva puede no contener `model.pkl`, `feature_stats.pkl`, `features_cache.pkl`, `audio/` o `manifest.csv`, aunque esos archivos existan en una copia local de trabajo.

Los archivos `.pkl` deben tratarse como artefactos generados localmente y no deben abrirse con `joblib` si provienen de una fuente no confiable.

---

## Limitaciones conocidas

- El backend exige audio con dos canales, pero no valida de forma explícita que el sample rate sea exactamente 8 kHz.
- La validación del frontend comprueba WAV y canales, pero el backend sigue siendo la autoridad final.
- La calidad del detector depende completamente del dataset, sus etiquetas y la similitud con llamadas reales.
- No se reportan métricas de evaluación (`precision`, `recall`, `F1`, `ROC-AUC`, matriz de confusión) como parte del entrenamiento actual.
- El split `train/test` se carga, pero el código de entrenamiento solo ajusta el modelo con `split == "train"`; no ejecuta una evaluación automática del split `test`.
- El breakdown representa rareza ponderada por importancia global; no es una explicación causal ni una garantía de que una categoría “causó” la decisión.
- La transcripción de Whisper puede ser costosa en CPU y el vocabulario semántico configurado es limitado.
- El webhook de WhatsApp es una simulación que imprime un payload; no envía mensajes reales ni detiene llamadas.
- CORS está abierto a cualquier origen en `app.main`.
- `app.model`/`explain_model.py` y `app.main` representan variantes diferentes del modelo y pueden sobrescribir el mismo `model.pkl`.
- No hay autenticación, rate limiting, control de tamaño máximo del payload ni protección específica para datos sensibles.
- El historial del frontend usa `localStorage`, por lo que no es un audit log centralizado ni compartido entre usuarios.

---

## Siguientes pasos

- Completar `requirements.txt` con todas las dependencias realmente importadas.
- Consolidar la API recomendada y retirar o aislar la variante experimental.
- Versionar el esquema de features junto con el modelo.
- Validar sample rate, duración, formato y tamaño máximo del audio.
- Añadir tests unitarios para VAD, extracción de 40 features, breakdown y endpoints.
- Añadir evaluación reproducible sobre un conjunto de prueba separado.
- Restringir CORS y proteger la API con autenticación.
- Sustituir la simulación de WhatsApp por un adaptador configurable y seguro.
- Evitar guardar audio sensible en logs o almacenamiento no cifrado.
- Añadir observabilidad, métricas de latencia y trazabilidad de decisiones.
- Configurar despliegue reproducible con Docker/CI.

---

## Resumen rápido

```bash
# Backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pandas joblib faster-whisper requests
uvicorn app.main:app --reload --port 8000

# En otra terminal: frontend
npm install
cp .env.example .env
npm run dev
```

Después abre <http://localhost:5173>, carga un WAV estéreo y ejecuta el análisis.
