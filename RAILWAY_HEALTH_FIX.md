# Railway startup / healthcheck fix

The worker no longer constructs the Whisper model at module import time.
`/health` is intentionally lightweight and returns immediately without loading Whisper or the local summary model.

## Recommended environment

```env
AI_API_KEY=YOUR_SHARED_SECRET

WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_BEAM_SIZE=3
WHISPER_VAD_FILTER=true

SUMMARY_MODEL=google/flan-t5-small
SUMMARY_MAX_INPUT_TOKENS=1800
SUMMARY_MAX_NEW_TOKENS=384
SUMMARY_NUM_BEAMS=2
SUMMARY_CHUNK_CHARS=7000

DATA_DIR=/var/data
WHISPER_MODEL_CACHE=/var/data/models
SUMMARY_MODEL_CACHE=/var/data/models
HF_HOME=/var/data/huggingface
HF_HUB_CACHE=/var/data/huggingface/hub

# Keep false initially. Enable only after health checks are stable if you want
# background model warmup after the web server has started.
MODEL_WARMUP=false
```

## Expected startup

Uvicorn starts the FastAPI app first. Railway can then call:

`GET /health`

without waiting for Whisper or the summary model.

The first transcription request will lazily load Whisper. The first summary request will lazily load the local summarization model.

## Validate locally

```bash
python -m py_compile app.py config.py worker.py summarizer.py
uvicorn app:app --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/health
```

## Railway

Keep the healthcheck path as `/health`.
The large image size is primarily caused by the ML stack (`torch`, `transformers`, `faster-whisper`, and related dependencies). This startup fix prevents model initialization from blocking healthchecks, but reducing the image size is a separate optimization.
