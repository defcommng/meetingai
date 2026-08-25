import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from config import AI_API_KEY, AUDIO_DIR, JOBS_DIR, MAX_UPLOAD_BYTES, OUTPUT_DIR, APP_NAME, ensure_directories
from worker import start_worker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("defcomm-ai")

app = FastAPI(title=APP_NAME, version="1.0.0")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_api_key(authorization: str | None) -> None:
    if not AI_API_KEY:
        raise HTTPException(status_code=500, detail="AI service API key is not configured")

    expected = f"Bearer {AI_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid AI service API key")


@app.on_event("startup")
def startup() -> None:
    ensure_directories()
    start_worker()
    logger.info("DefComm AI service started")


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "service": APP_NAME}


@app.post("/v1/transcriptions")
async def create_transcription(
    meeting_id: Annotated[str, Form()],
    recording_id: Annotated[str, Form()],
    metadata: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()],
    authorization: Annotated[str | None, Header()] = None,
):
    verify_api_key(authorization)

    if not files:
        raise HTTPException(status_code=400, detail="At least one audio file is required")

    try:
        metadata_items = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="metadata must be valid JSON")

    if not isinstance(metadata_items, list):
        raise HTTPException(status_code=400, detail="metadata must be a JSON array")

    metadata_by_filename = {
        item["filename"]: item
        for item in metadata_items
        if isinstance(item, dict)
        and isinstance(item.get("filename"), str)
        and isinstance(item.get("participant_id"), str)
    }

    job_id = str(uuid.uuid4())
    job_audio_dir = AUDIO_DIR / job_id
    job_output_dir = OUTPUT_DIR / job_id
    job_audio_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    tracks = []

    try:
        for upload in files:
            original_filename = upload.filename or f"{uuid.uuid4()}.mkv"
            metadata_item = metadata_by_filename.get(original_filename)
            if not metadata_item:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing participant metadata for {original_filename}",
                )

            safe_filename = Path(original_filename).name
            destination = job_audio_dir / safe_filename
            file_bytes = 0

            with destination.open("wb") as output_file:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    file_bytes += len(chunk)
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Transcription upload is too large")
                    output_file.write(chunk)

            await upload.close()

            tracks.append(
                {
                    "path": str(destination),
                    "filename": original_filename,
                    "participant_id": metadata_item["participant_id"],
                    "speaker_name": metadata_item.get("speaker_name"),
                    "bytes": file_bytes,
                }
            )

    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Failed to store transcription upload")
        raise HTTPException(status_code=500, detail=str(error))

    job = {
        "job_id": job_id,
        "meeting_id": meeting_id,
        "recording_id": recording_id,
        "created_at": utc_now(),
        "status": "queued",
        "tracks": tracks,
        "output_dir": str(job_output_dir),
        "transcript_path": None,
        "text_transcript_path": None,
        "error": None,
    }

    job_path = JOBS_DIR / f"{job_id}.json"
    with job_path.open("w", encoding="utf-8") as file:
        json.dump(job, file, indent=2, ensure_ascii=False)

    return {
        "job_id": job_id,
        "status": "queued",
        "meeting_id": meeting_id,
        "recording_id": recording_id,
    }


@app.get("/v1/transcriptions/{job_id}")
def get_transcription_status(
    job_id: str,
    authorization: str | None = Header(default=None),
):
    verify_api_key(authorization)
    job_path = JOBS_DIR / f"{job_id}.json"
    if not job_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    with job_path.open("r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/v1/transcriptions/{job_id}/transcript")
def download_transcript(
    job_id: str,
    authorization: str | None = Header(default=None),
):
    verify_api_key(authorization)
    job_path = JOBS_DIR / f"{job_id}.json"
    if not job_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    with job_path.open("r", encoding="utf-8") as file:
        job = json.load(file)

    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Transcription is not completed")

    transcript_path = Path(job["transcript_path"])
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file missing")

    return FileResponse(
        path=transcript_path,
        media_type="application/json",
        filename="transcript.json",
    )


@app.get("/v1/transcriptions/{job_id}/text")
def download_text_transcript(
    job_id: str,
    authorization: str | None = Header(default=None),
):
    verify_api_key(authorization)
    job_path = JOBS_DIR / f"{job_id}.json"
    if not job_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    with job_path.open("r", encoding="utf-8") as file:
        job = json.load(file)

    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Transcription is not completed")

    transcript_path = Path(job["text_transcript_path"])
    if not transcript_path.exists():
        raise HTTPException(status_code=404, detail="Transcript file missing")

    return FileResponse(
        path=transcript_path,
        media_type="text/plain",
        filename="transcript.txt",
    )
