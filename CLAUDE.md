# transcribe-podcast — Guía de desarrollo para Claude

## Descripción del proyecto

Herramienta CLI en Python que transcribe podcasts `.mp3` localmente con `openai-whisper`
y genera resúmenes en Markdown usando un LLM a través de OpenRouter.
Episodios de más de 1 hora usan LangChain map-reduce automáticamente.

## Comandos esenciales

```bash
# Instalar en modo desarrollo
pip3 install -e ".[dev]"

# Ejecutar la herramienta
transcribe-podcast
transcribe-podcast --input-dir ~/podcasts --output-dir ~/resumenes
transcribe-podcast --whisper-model small
transcribe-podcast --json

# Linting y formato (SIEMPRE antes de commitear)
ruff check . --fix
ruff format .

# Tests
pytest
pytest --cov=transcribe_podcast
```

## Arquitectura

```
cli.py → processor.py → transcriber.py   (openai-whisper, local)
                      ↘ summarizer.py    (LangChain → OpenRouter)
                           ├── corto (<1h): ChatOpenAI single call
                           └── largo (≥1h): RecursiveCharacterTextSplitter + map_reduce chain
```

## Archivos clave

| Archivo | Responsabilidad |
|---------|----------------|
| `src/transcribe_podcast/cli.py` | Entry point `main()`, argparse, bucle batch, output human/JSON |
| `src/transcribe_podcast/config.py` | `AppConfig` dataclass, `load_config()` — carga `.env` y args CLI |
| `src/transcribe_podcast/transcriber.py` | `PodcastFile`, `Transcription`, `discover_files()`, `transcribe()` |
| `src/transcribe_podcast/summarizer.py` | `Summary`, `write_summary()`, `build_llm()`, `summarise()`, `_summarise_long()` |
| `src/transcribe_podcast/processor.py` | `ProcessingResult`, `process_file()` — orquesta y captura errores |

## Estructura de carpetas

```
src/transcribe_podcast/   # Paquete principal
tests/unit/               # Tests unitarios (con mocks de whisper/LLM)
tests/integration/        # Tests de integración
input/                    # Archivos .mp3 de entrada (gitignored)
output/                   # Resúmenes .md generados (gitignored)
.env                      # Credenciales (gitignored, nunca commitear)
.env.example              # Plantilla segura de variables de entorno
pyproject.toml            # Dependencias y entry point
```

## Variables de entorno requeridas (`.env`)

```dotenv
OPENROUTER_API_KEY=sk-or-...        # requerido
OPENROUTER_MODEL=openai/gpt-4o-mini # requerido
WHISPER_MODEL=base                  # opcional, default: base
INPUT_DIR=./input                   # opcional
OUTPUT_DIR=./output                 # opcional
```

## Convenciones de código

- Python 3.11+, `from __future__ import annotations` en todos los módulos
- `ruff` — line-length 100, target py311, reglas E/F/W/I
- Dataclasses para todas las entidades de dominio (no dicts sueltos)
- Imports de whisper y langchain chains dentro de las funciones que los usan (lazy import)
- Errores de configuración → `sys.exit(1)` con mensaje descriptivo en stderr
- Errores en archivos individuales → capturados en `ProcessingResult`, no propagan

## Lógica clave

**Detección de episodio largo:**
```python
duration_s = segments[-1]["end"] if segments else 0.0
is_long = duration_s >= 3600  # 1 hora en segundos
```

**Dispatch de resumen:**
```python
if transcription.is_long:
    _summarise_long(transcription, config)   # map-reduce
else:
    _summarise_short(transcription, config)  # single LLM call
```

**Naming de salida:**
```python
output_path = config.output_dir / (podcast_file.stem + ".md")
```

## Dependencias

```
openai-whisper>=20231117   # transcripción local
langchain>=0.2             # chains de resumen
langchain-openai>=0.1      # ChatOpenAI → OpenRouter
langchain-text-splitters>=0.2  # RecursiveCharacterTextSplitter
python-dotenv>=1.0         # carga .env
```

## Configuración OpenRouter

```python
ChatOpenAI(
    model=config.model,
    api_key=config.api_key,
    base_url="https://openrouter.ai/api/v1",
)
```

## Exit codes

| Código | Causa |
|--------|-------|
| `0` | Éxito total (o carpeta vacía) |
| `1` | Error de configuración |
| `2` | Fallo parcial en batch |
