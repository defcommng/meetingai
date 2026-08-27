import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from config import (
    AI_API_KEY,
    APP_NAME,
    AUDIO_DIR,
    JOBS_DIR,
    MODEL_WARMUP,
    OUTPUT_DIR,
    MAX_UPLOAD_BYTES,
    SUMMARY_ENABLED,
    ensure_directories,
)
from summarizer import SUMMARY_MODEL, _load_model
from worker import get_whisper_model, start_worker

logger = logging.getLogger("defcomm-ai")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=APP_NAME, version="2.0.0")


def verify_api_key(authorization: str | None) -> None:
    if not AI_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    supplied = authorization.removeprefix("Bearer ")
    if supplied != AI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid AI service API key")


@app.on_event("startup")
def startup() -> None:
    ensure_directories()
    start_worker()

    if MODEL_WARMUP:
        def warm_models() -> None:
            try:
                get_whisper_model()
                if SUMMARY_ENABLED:
                    _load_model()
                logger.info("AI models warmed successfully")
            except Exception:
                logger.exception("AI model warmup failed; lazy loading remains enabled")

        threading.Thread(
            target=warm_models,
            name="defcomm-ai-model-warmup",
            daemon=True,
        ).start()

    logger.info("DefComm AI service started")


@app.get("/health")
def health() -> dict:
    from worker import _model as whisper_model
    from summarizer import _model as summary_model

    return {
        "ok": True,
        "service": APP_NAME,
        "version": "2.0.0",
        "mode": "post-meeting",
        "summary_enabled": SUMMARY_ENABLED,
        "summary_model": SUMMARY_MODEL,
        "whisper_loaded": whisper_model is not None,
        "summary_loaded": summary_model is not None,
    }


@app.post("/v1/transcriptions")
async def create_transcription(
    request: Request,
    meeting_id: Annotated[str, Form()],
    recording_id: Annotated[str, Form()],
    metadata: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()],
):
    verify_api_key(request.headers.get("authorization"))

    try:
        metadata_items = json.loads(metadata)
        if not isinstance(metadata_items, list):
            raise ValueError
    except Exception as error:
        raise HTTPException(status_code=400, detail="metadata must be a JSON array") from error

    metadata_by_filename = {
        item["filename"]: item
        for item in metadata_items
        if isinstance(item, dict)
        and "filename" in item
        and "participant_id" in item
    }

    job_id = str(uuid.uuid4())
    job_audio_dir = AUDIO_DIR / job_id
    job_output_dir = OUTPUT_DIR / job_id
    job_audio_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    tracks: list[dict] = []
    total_bytes = 0

    try:
        for upload in files:
            original_filename = upload.filename or f"{uuid.uuid4()}.mkv"
            item = metadata_by_filename.get(original_filename)
            if not item:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing participant metadata for {original_filename}",
                )

            safe_filename = Path(original_filename).name
            destination = job_audio_dir / safe_filename

            written = 0
            with destination.open("wb") as output:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Upload exceeds configured size limit")
                    output.write(chunk)
            await upload.close()

            if written == 0:
                raise HTTPException(status_code=400, detail=f"Empty media file: {original_filename}")

            tracks.append({
                "path": str(destination),
                "filename": original_filename,
                "participant_id": str(item["participant_id"]),
                "speaker_name": item.get("speaker_name") or item.get("participant_name"),
                "offset_ms": int(item.get("offset_ms") or 0),
            })

    except Exception:
        # Do not leave half-created jobs around when upload validation fails.
        for path in job_audio_dir.glob("*"):
            path.unlink(missing_ok=True)
        job_audio_dir.rmdir()
        job_output_dir.rmdir()
        raise

    if not tracks:
        raise HTTPException(status_code=400, detail="At least one media file is required")

    job = {
        "job_id": job_id,
        "meeting_id": meeting_id,
        "recording_id": recording_id,
        "created_at": None,
        "status": "queued",
        "stage": "queued",
        "tracks": tracks,
        "output_dir": str(job_output_dir),
        "transcript_path": None,
        "text_transcript_path": None,
        "summary_path": None,
        "error": None,
    }

    job_path = JOBS_DIR / f"{job_id}.json"
    job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")

    return {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "meeting_id": meeting_id,
        "recording_id": recording_id,
    }


@app.get("/v1/transcriptions/{job_id}")
def get_transcription_status(job_id: str):
    path = JOBS_DIR / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/v1/transcriptions/{job_id}/transcript")
def download_transcript(job_id: str):
    job_path = JOBS_DIR / f"{job_id}.json"
    if not job_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    job = json.loads(job_path.read_text(encoding="utf-8"))
    path_value = job.get("transcript_path")
    if not path_value:
        raise HTTPException(status_code=409, detail="Transcript is not available")

    path = Path(path_value)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript file missing")

    return FileResponse(path=path, media_type="application/json", filename="transcript.json")


@app.get("/v1/transcriptions/{job_id}/summary")
def download_summary(job_id: str):
    job_path = JOBS_DIR / f"{job_id}.json"
    if not job_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    job = json.loads(job_path.read_text(encoding="utf-8"))
    path_value = job.get("summary_path")
    if not path_value:
        raise HTTPException(status_code=409, detail="Summary is not available")

    path = Path(path_value)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Summary file missing")

    return FileResponse(path=path, media_type="application/json", filename="summary.json")


@app.post("/v1/transcriptions/{job_id}/retry")
def retry_failed_transcription(job_id: str, request: Request):
    verify_api_key(request.headers.get("authorization"))

    path = JOBS_DIR / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    job = json.loads(path.read_text(encoding="utf-8"))
    if job.get("status") not in {"failed", "completed"}:
        raise HTTPException(status_code=409, detail=f"Cannot retry job in state {job.get('status')}")

    # Retry the exact same uploaded files. This is useful for summary-only
    # failures too, because the worker can rebuild transcript and summary safely.
    job.update({
        "status": "queued",
        "stage": "queued",
        "error": None,
        "retry_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "completed_at": None,
        "summary_path": None,
    })
    path.write_text(json.dumps(job, indent=2), encoding="utf-8")

    return {"job_id": job_id, "status": "queued", "stage": "queued"}
