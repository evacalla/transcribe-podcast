# transcribe-podcast

Herramienta de línea de comandos que transcribe archivos de audio de podcasts (`.mp3`) de forma local usando **Whisper**.

> **Nota:** La herramienta ahora genera transcripciones raw en texto plano. El módulo de resumen (`summarizer.py`) sigue disponible para uso standalone con OpenRouter.

## Cómo funciona

```
.mp3  →  [Whisper local]  →  transcripción .md
```

1. Lee la configuración de un archivo `.env` (opcional)
2. Escanea una carpeta de entrada en busca de archivos `.mp3`
3. Transcribe cada archivo **localmente** con `openai-whisper` (sin enviar audio a ninguna API)
4. Guarda la transcripción en un archivo `.md` con el mismo nombre que el `.mp3`

### Episodios largos (más de 1 hora)

La herramienta detecta automáticamente episodios de más de 1 hora. Esta información está disponible en el output para procesamiento posterior.

## Requisitos

- Python 3.11+
- Archivos `.mp3` para procesar
- (Opcional) Cuenta en [OpenRouter](https://openrouter.ai) si quieres usar el módulo de resumen standalone

## Instalación

```bash
pip install -e ".[dev]"
```

## Configuración

```bash
cp .env.example .env
```

Editar `.env`:

```dotenv
WHISPER_MODEL=base    # tiny | base | small | medium | large
INPUT_DIR=./input     # carpeta donde están los .mp3
OUTPUT_DIR=./output   # carpeta donde se escriben las transcripciones .md
WHISPER_LANGUAGE=es   # idioma fijo (ej. es, en) — omitir para detección automática
WHISPER_FP16=false    # false → fuerza fp32; true → fuerza fp16; omitir → automático
```

> **Para uso del módulo de resumen standalone:**
> ```dotenv
> OPENROUTER_API_KEY=sk-or-...        # requerido
> OPENROUTER_MODEL=openai/gpt-4o-mini # requerido
> ```

## Uso

```bash
# Transcribir todos los .mp3 en ./input → archivos .md en ./output
transcribe-podcast

# Carpetas personalizadas
transcribe-podcast --input-dir ~/podcasts --output-dir ~/transcripciones

# Modelo Whisper más preciso (más lento)
transcribe-podcast --whisper-model small

# Forzar fp32 (mayor precisión en algunos dispositivos, especialmente Apple Silicon)
transcribe-podcast --no-fp16

# Forzar fp16 explícitamente
transcribe-podcast --fp16

# Idioma fijo (evita el paso de detección automática)
transcribe-podcast --language es

# Salida JSON (útil para scripts)
transcribe-podcast --json
```

### Precisión numérica (`--fp16` / `--no-fp16`)

Por defecto la herramienta usa **fp16 solo en CUDA** y **fp32 en MPS y CPU**. Esto evita problemas de NaN que pueden ocurrir con fp16 en Apple Silicon. Para forzar un modo específico:

```bash
transcribe-podcast --no-fp16   # fuerza fp32
transcribe-podcast --fp16      # fuerza fp16 (puede causar problemas en MPS)
# equivalente en .env:  WHISPER_FP16=false
```

### Salida por consola (modo por defecto)

La herramienta muestra logs detallados del proceso:

```
[INIT] Starting podcast transcription batch
[CONFIG] Input: /home/user/podcasts/input
[CONFIG] Output: /home/user/podcasts/output
[CONFIG] Whisper Model: base

[1/3] Processing: episodio-42.mp3
      Loading Whisper 'base' on mps...
      Transcribing audio...
      [DURATION] Episode duration: 45.3 minutes
      [TIME] Processed in 123.5s
      Saved: /home/user/podcasts/output/episodio-42.md

[2/3] Processing: entrevista-larga.mp3
      Loading Whisper 'base' on mps...
      Transcribing audio...
      [DURATION] Episode duration: 65.2 minutes
      Long episode detected
      [TIME] Processed in 189.2s
      Saved: /home/user/podcasts/output/entrevista-larga.md

[3/3] Processing: archivo-roto.mp3
      Loading Whisper 'base' on mps...
      ERROR: Could not read audio file

[DONE] Batch complete: 2 succeeded, 1 failed.
```

### Salida JSON (`--json`)

```json
{
  "total": 3,
  "succeeded": 2,
  "failed": 1,
  "results": [
    { "file": "episodio-42.mp3", "status": "success", "output": "output/episodio-42.md", "duration_min": 45.3 },
    { "file": "entrevista-larga.mp3", "status": "success", "output": "output/entrevista-larga.md", "duration_min": 65.2 },
    { "file": "archivo-roto.mp3", "status": "error", "error": "Could not read audio file" }
  ]
}
```

### Códigos de salida

| Código | Significado |
|--------|-------------|
| `0` | Todos los archivos procesados correctamente (o carpeta vacía) |
| `1` | Error de configuración (credenciales faltantes, carpeta inexistente) |
| `2` | Fallo parcial — al menos un archivo falló |

## Estructura del proyecto

```
transcribe-podcast/
├── src/
│   └── transcribe_podcast/
│       ├── cli.py          # Entry point CLI (argparse, bucle batch, output)
│       ├── config.py       # Carga .env + args, valida, devuelve AppConfig
│       ├── transcriber.py  # Integración openai-whisper, detección de duración
│       ├── summarizer.py   # (Opcional) Resumen via OpenRouter standalone
│       └── processor.py    # Orquestación por archivo con manejo de errores
├── tests/
│   ├── unit/
│   └── integration/
├── input/                  # Poner aquí los archivos .mp3 (gitignored)
├── output/                 # Las transcripciones .md se escriben aquí (gitignored)
├── .env.example            # Plantilla de variables de entorno
└── pyproject.toml          # Dependencias y configuración del proyecto
```

## Dependencias principales

| Paquete | Uso |
|---------|-----|
| `openai-whisper` | Transcripción local de audio (sin API externa) |
| `openai` | Cliente LLM para el módulo de resumen standalone |
| `python-dotenv` | Carga de variables de entorno desde `.env` |

## Modelos Whisper disponibles

| Modelo | RAM aprox. | Velocidad | Precisión |
|--------|-----------|-----------|-----------|
| `tiny` | ~390 MB | Muy rápido | Básica |
| `base` | ~1.4 GB | Rápido | Buena (**por defecto**) |
| `small` | ~2.3 GB | Moderado | Muy buena |
| `medium` | ~5 GB | Lento | Alta |
| `large` | ~10 GB | Muy lento | Máxima |

## Desarrollo

```bash
# Linting
ruff check .

# Formateo
ruff format .

# Tests
pytest
```

---

## Uso del módulo de resumen (standalone)

El módulo `summarizer.py` puede usarse independientemente para generar resúmenes de transcripciones ya existentes:

```python
from transcribe_podcast.summarizer import summarise
from transcribe_podcast.transcriber import Transcription
from transcribe_podcast.config import AppConfig

# Cargar transcripción
transcription = Transcription(text="...", duration_s=3600, is_long=True)

# Configurar (requiere OPENROUTER_API_KEY y OPENROUTER_MODEL en .env)
config = AppConfig(
    api_key="sk-or-...",
    model="openai/gpt-4o-mini",
    # ... otros campos
)

# Generar resumen
summary_text, was_chunked = summarise(transcription, config)
```

Para episodios largos (>1h), el resumen usa map-reduce con división automática de texto.
