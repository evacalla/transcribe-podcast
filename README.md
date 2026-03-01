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
WHISPER_MODEL=base   # tiny | base | small | medium | large
INPUT_DIR=./input    # carpeta donde están los .mp3
OUTPUT_DIR=./output  # carpeta donde se escriben los .md
```

## Uso

```bash
# Procesar todos los .mp3 en ./input → resúmenes en ./output
transcribe-podcast

# Carpetas personalizadas
transcribe-podcast --input-dir ~/podcasts --output-dir ~/resumenes

# Modelo Whisper más preciso (más lento)
transcribe-podcast --whisper-model small

# Salida JSON (útil para scripts)
transcribe-podcast --json
```

### Salida por consola (modo por defecto)

```
[1/3] Processing: episodio-42.mp3
      Saved: /ruta/output/episodio-42.md

[2/3] Processing: entrevista-larga.mp3
      Long episode detected — splitting into chunks
      Saved: /ruta/output/entrevista-larga.md

[3/3] Processing: archivo-roto.mp3
      ERROR: Could not read audio file

Done: 2 succeeded, 1 failed.
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
| `0` | Todos los archivos procesados correctamente |
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
