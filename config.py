import os
from pathlib import Path


APP_NAME = os.getenv(
    "APP_NAME",
    "DefComm AI",
)

DATA_DIR = Path(
    os.getenv(
        "DATA_DIR",
        "/tmp/defcomm-ai",
    )
).resolve()

JOBS_DIR = DATA_DIR / "jobs"
AUDIO_DIR = DATA_DIR / "audio"
OUTPUT_DIR = DATA_DIR / "outputs"


WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "small",
)

WHISPER_DEVICE = os.getenv(
    "WHISPER_DEVICE",
    "cpu",
)

WHISPER_COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE",
    "int8",
)

WHISPER_BEAM_SIZE = int(
    os.getenv(
        "WHISPER_BEAM_SIZE",
        "5",
    )
)

AI_API_KEY = os.getenv(
    "AI_API_KEY",
    "",
)

WORKER_POLL_SECONDS = float(
    os.getenv(
        "WORKER_POLL_SECONDS",
        "2",
    )
)


def ensure_directories() -> None:
    for directory in (
        DATA_DIR,
        JOBS_DIR,
        AUDIO_DIR,
        OUTPUT_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )