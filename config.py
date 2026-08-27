import os
from pathlib import Path

APP_NAME = os.getenv("APP_NAME", "DefComm AI")
DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/defcomm-ai")).resolve()
JOBS_DIR = DATA_DIR / "jobs"
AUDIO_DIR = DATA_DIR / "audio"
OUTPUT_DIR = DATA_DIR / "outputs"
MODEL_DIR = DATA_DIR / "models"

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "3"))
WHISPER_VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "true").lower() == "true"
AI_API_KEY = os.getenv("AI_API_KEY", "")
WORKER_POLL_SECONDS = float(os.getenv("WORKER_POLL_SECONDS", "2"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
HF_HOME = os.getenv("HF_HOME", "/tmp/huggingface")
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE", "/tmp/huggingface/hub")


def ensure_directories() -> None:
    for directory in (DATA_DIR, JOBS_DIR, AUDIO_DIR, OUTPUT_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)

MODEL_WARMUP = os.getenv("MODEL_WARMUP", "false").lower() == "true"
