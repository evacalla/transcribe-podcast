# transcribe-podcast

Herramienta de línea de comandos que transcribe archivos de audio de podcasts (`.mp3`) de forma local usando **Whisper** y genera un resumen escrito usando un modelo de lenguaje a través de **OpenRouter**.

## Cómo funciona

```
.mp3  →  [Whisper local]  →  texto completo  →  [LLM via OpenRouter]  →  resumen .md
```

1. Lee las credenciales de un archivo `.env`
2. Escanea una carpeta de entrada en busca de archivos `.mp3`
3. Transcribe cada archivo **localmente** con `openai-whisper` (sin enviar audio a ninguna API)
4. Genera un resumen usando el modelo configurado en OpenRouter
5. Guarda el resumen en un archivo `.md` con el mismo nombre que el `.mp3`

### Episodios largos (más de 1 hora)

Cuando la duración supera 1 hora, el texto de la transcripción puede exceder el límite de contexto del modelo. En ese caso el proceso usa **LangChain map-reduce**:

```
transcripción  →  [divide en chunks]  →  resumen parcial x N  →  [fusiona]  →  resumen final
```

Esto ocurre de forma automática, sin configuración adicional.

## Requisitos

- Python 3.11+
- Cuenta en [OpenRouter](https://openrouter.ai) con una API key
- Archivos `.mp3` para procesar

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
OPENROUTER_API_KEY=sk-or-...        # requerido
OPENROUTER_MODEL=openai/gpt-4o-mini # requerido — modelo a usar para el resumen
```

Variables opcionales (también se pueden pasar como flags de CLI):

```dotenv
WHISPER_MODEL=base    # tiny | base | small | medium | large
INPUT_DIR=./input     # carpeta donde están los .mp3
OUTPUT_DIR=./output   # carpeta donde se escriben los .md
WHISPER_LANGUAGE=es   # idioma fijo (ej. es, en) — omitir para detección automática
WHISPER_FP16=false    # false → fuerza fp32; true → fuerza fp16; omitir → automático
```

### Delay entre episodios

Para evitar exceder los rate limits de OpenRouter, la herramienta incluye un **delay automático** entre episodios:
- **1 segundo por cada minuto de audio** del episodio procesado
- Ejemplo: un episodio de 45 minutos → espera 45 segundos antes de continuar
- Esto ocurre después de guardar el resumen, antes de pasar al siguiente archivo

## Uso

```bash
# Procesar todos los .mp3 en ./input → resúmenes en ./output
transcribe-podcast

# Carpetas personalizadas
transcribe-podcast --input-dir ~/podcasts --output-dir ~/resumenes

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

Por defecto Whisper usa **fp16** en GPU (MPS/CUDA) y **fp32** en CPU. En algunos dispositivos — particularmente Apple Silicon — fp32 produce mejores transcripciones. Usa `--no-fp16` para forzarlo:

```bash
transcribe-podcast --no-fp16
# equivalente en .env:  WHISPER_FP16=false
```

### Salida por consola (modo por defecto)

La herramienta muestra logs detallados del proceso:

```
[INIT] Starting podcast transcription batch
[CONFIG] Input: /home/user/podcasts/input
[CONFIG] Output: /home/user/podcasts/output
[CONFIG] LLM Model: openai/gpt-4o-mini
[CONFIG] Whisper Model: base

[1/3] Processing: episodio-42.mp3
      [START] Processing file: episodio-42.mp3
      [MODEL] Using LLM model: openai/gpt-4o-mini
      [WHISPER] Using Whisper model: base
      Loading Whisper model 'base' on mps...
      Transcribing audio...
      [DURATION] Episode duration: 45.3 minutes
      [SUMMARY] Generating summary...
      [LLM] Building LLM with model: openai/gpt-4o-mini
      [LLM] Sending request to OpenRouter...
      [LLM] Response received
      [SUMMARY] Summary generated (chunked: False)
      [DONE] Summary saved to: /home/user/podcasts/output/episodio-42.md
      [DELAY] Waiting 45 seconds (45.3 min)...

[2/3] Processing: entrevista-larga.mp3
      [START] Processing file: entrevista-larga.mp3
      [MODEL] Using LLM model: openai/gpt-4o-mini
      [WHISPER] Using Whisper model: base
      Loading Whisper model 'base' on mps...
      Transcribing audio...
      [DURATION] Episode duration: 65.2 minutes
      [SUMMARY] Generating summary...
      [LLM] Building LLM with model: openai/gpt-4o-mini
      [LLM] Splitting into 4 chunks for map-reduce...
      [LLM] Starting map-reduce summarization (this may take a while)...
      [LLM] Map-reduce complete
      [SUMMARY] Summary generated (chunked: True)
      [DONE] Summary saved to: /home/user/podcasts/output/entrevista-larga.md
      [DELAY] Waiting 65 seconds (65.2 min)...

[3/3] Processing: archivo-roto.mp3
      [START] Processing file: archivo-roto.mp3
      [MODEL] Using LLM model: openai/gpt-4o-mini
      [WHISPER] Using Whisper model: base
      Loading Whisper model 'base' on mps...
      [ERROR] Could not read audio file

[DONE] Batch complete: 2 succeeded, 1 failed.
```

### Salida JSON (`--json`)

```json
{
  "total": 3,
  "succeeded": 2,
  "failed": 1,
  "results": [
    { "file": "episodio-42.mp3", "status": "success", "output": "output/episodio-42.md", "chunked": false },
    { "file": "entrevista-larga.mp3", "status": "success", "output": "output/entrevista-larga.md", "chunked": true },
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
│       ├── summarizer.py   # Resumen via OpenRouter (single-pass o map-reduce)
│       └── processor.py    # Orquestación por archivo con manejo de errores
├── tests/
│   ├── unit/
│   └── integration/
├── input/                  # Poner aquí los archivos .mp3 (gitignored)
├── output/                 # Los resúmenes .md se escriben aquí (gitignored)
├── .env.example            # Plantilla de variables de entorno
└── pyproject.toml          # Dependencias y configuración del proyecto
```

## Dependencias principales

| Paquete | Uso |
|---------|-----|
| `openai-whisper` | Transcripción local de audio (sin API externa) |
| `langchain-openai` | Cliente LLM compatible con OpenRouter |
| `langchain-text-splitters` | División de texto para episodios largos |
| `langchain` | Pipeline map-reduce de resumen |
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
