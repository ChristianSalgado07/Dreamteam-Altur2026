"""
Generates the external synthetic-speech samples for the external
validation experiment, using two open-source/free TTS systems that are
architecturally different from each other AND from whatever produced
the original dataset's Channel-0 synthetic recordings (a fluent,
natural-sounding neural TTS/voice-cloning system, per the transcripts
in outputs/transcripts/):

  - "gtts_google": Google Translate's TTS endpoint via the `gTTS`
    Python package (a real, modern, widely-used web TTS service).
  - "espeak_ng": espeak-ng, a classic formant/rule-based synthesizer —
    deliberately very different in kind from both the original
    dataset's synthetic voice and from gTTS, to give a second, clearly
    distinct generator for the generator-level analysis.

Text content: 24 generic, conversational Spanish sentences (no PII, not
copied from any copyrighted source, written for this experiment) with
varying length for a spread of clip durations. Same 24 sentences used
for both generators so any speaking-rate/duration differences reflect
the generator, not the text.

Run with: python external_data/generate_synthetic.py
"""
import csv
import subprocess
from pathlib import Path

from gtts import gTTS

EXTERNAL_DIR = Path(__file__).resolve().parent
GTTS_DIR = EXTERNAL_DIR / "raw" / "synthetic_gtts"
ESPEAK_DIR = EXTERNAL_DIR / "raw" / "synthetic_espeak"
MANIFEST_CSV = EXTERNAL_DIR / "manifests" / "synthetic_generation_manifest.csv"

SENTENCES = [
    "Hola, buenos días, ¿en qué le puedo ayudar hoy?",
    "Quisiera revisar el estado de mi cuenta, por favor.",
    "El clima en la ciudad ha estado muy variable esta semana.",
    "Me gustaría reservar una mesa para cuatro personas esta noche.",
    "El tren sale de la estación central a las nueve de la mañana.",
    "Por favor, envíeme una copia del contrato por correo electrónico.",
    "El proyecto se entregará la próxima semana según lo planeado.",
    "¿Podría repetir su número de referencia, por favor?",
    "La reunión ha sido reprogramada para el jueves por la tarde.",
    "Necesito actualizar mi dirección en el sistema.",
    "El paquete llegará entre dos y cinco días hábiles.",
    "Gracias por su paciencia mientras resolvemos este inconveniente.",
    "El restaurante está ubicado cerca de la plaza principal.",
    "¿Cuál es el horario de atención los fines de semana?",
    "La biblioteca cierra a las ocho de la noche entre semana.",
    "Le recomiendo revisar los términos y condiciones antes de continuar.",
    "El vuelo tiene una escala de dos horas en la capital.",
    "Estamos trabajando para resolver el problema lo antes posible.",
    "¿Me puede confirmar su nombre completo y fecha de nacimiento?",
    "El equipo técnico visitará su domicilio mañana por la mañana.",
    "Recuerde traer una identificación válida a la cita.",
    "El pago se procesará en un plazo máximo de tres días.",
    "Muchas gracias por comunicarse con nosotros, que tenga buen día.",
    "¿Desea que le enviemos una notificación cuando esté listo?",
]


def generate_gtts():
    rows = []
    for i, text in enumerate(SENTENCES):
        clip_id = f"ext_gtts_{i:03d}"
        mp3_path = GTTS_DIR / f"{clip_id}.mp3"
        gTTS(text=text, lang="es").save(str(mp3_path))
        rows.append({"external_id": clip_id, "generator_id": "gtts_google", "text": text, "raw_file": str(mp3_path)})
        print(f"gTTS: {clip_id}")
    return rows


def generate_espeak():
    rows = []
    for i, text in enumerate(SENTENCES):
        clip_id = f"ext_espeak_{i:03d}"
        wav_path = ESPEAK_DIR / f"{clip_id}.wav"
        subprocess.run(
            ["espeak-ng", "-v", "es", "-w", str(wav_path), text],
            check=True, capture_output=True,
        )
        rows.append({"external_id": clip_id, "generator_id": "espeak_ng", "text": text, "raw_file": str(wav_path)})
        print(f"espeak-ng: {clip_id}")
    return rows


def run():
    GTTS_DIR.mkdir(parents=True, exist_ok=True)
    ESPEAK_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_gtts() + generate_espeak()
    with open(MANIFEST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["external_id", "generator_id", "text", "raw_file"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {MANIFEST_CSV}")


if __name__ == "__main__":
    run()
