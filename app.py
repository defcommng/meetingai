import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from config import (
    AI_API_KEY,
    APP_NAME,
    AUDIO_DIR,
    JOBS_DIR,
    OUTPUT_DIR,
    MAX_UPLOAD_BYTES,
    SUMMARY_ENABLED,
    WORKER_STALE_SECONDS,
    ensure_directories,
)
from summarizer import SUMMARY_MODEL
from worker import start_worker, worker_is_alive, read_json, write_json

logger = logging.getLogger("defcomm-ai")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=APP_NAME, version="2.1.0")


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

    logger.info("DefComm AI service started")


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "service": APP_NAME,
        "version": "2.1.0",
        "mode": "post-meeting",
        "summary_enabled": SUMMARY_ENABLED,
        "summary_model": SUMMARY_MODEL,
        "worker_alive": worker_is_alive(),
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
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
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
    return read_json(path)


@app.get("/v1/transcriptions/{job_id}/transcript")
def download_transcript(job_id: str):
    job_path = JOBS_DIR / f"{job_id}.json"
    if not job_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    job = read_json(job_path)
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

    job = read_json(job_path)
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

    job = read_json(path)
    state = job.get("status")

    # Failed/completed jobs can always be retried. A processing job may only be
    # retried when it is stale; this prevents two workers from transcribing the
    # same files concurrently while still recovering jobs abandoned by crashes.
    if state == "processing":
        started_at = job.get("started_at")
        if started_at:
            try:
                parsed = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                age = datetime.now(timezone.utc).timestamp() - parsed.timestamp()
                if age < WORKER_STALE_SECONDS:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Job is still processing (age={int(age)}s); retry is allowed after {WORKER_STALE_SECONDS}s",
                    )
            except ValueError:
                pass
        else:
            raise HTTPException(status_code=409, detail="Cannot retry processing job without a start timestamp")
    elif state not in {"failed", "completed", "queued"}:
        raise HTTPException(status_code=409, detail=f"Cannot retry job in state {state}")

    now = datetime.now(timezone.utc).isoformat()
    job.update({
        "status": "queued",
        "stage": "queued",
        "error": None,
        "retry_at": now,
        "updated_at": now,
        "started_at": None,
        "completed_at": None,
        "summary_path": None,
        "transcript_path": None,
        "text_transcript_path": None,
    })
    write_json(path, job)

    return {"job_id": job_id, "status": "queued", "stage": "queued"}

