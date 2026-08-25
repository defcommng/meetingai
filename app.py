import json
import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from config import (
    AI_API_KEY,
    AUDIO_DIR,
    JOBS_DIR,
    OUTPUT_DIR,
    APP_NAME,
    ensure_directories,
)
from worker import start_worker, transcribe_audio

logger = logging.getLogger("defcomm-ai")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=APP_NAME, version="1.1.0")


def verify_api_key(authorization: str | None = None):
    if not AI_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if authorization.removeprefix("Bearer ") != AI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid AI service API key")


@app.on_event("startup")
def startup() -> None:
    ensure_directories()
    start_worker()
    logger.info("DefComm AI service started")


@app.get("/health")
def health():
    return {"ok": True, "service": APP_NAME}


# Actual implementation using a Request keeps multipart and Authorization clean.
from fastapi import Request

@app.post("/v1/live/transcribe")
async def live_transcribe(
    request: Request,
    audio: Annotated[UploadFile, File()],
    meeting_id: Annotated[str, Form()],
    participant_id: Annotated[str, Form()],
    speaker_name: Annotated[str, Form()] = "Participant",
    sequence: Annotated[int, Form()] = 0,
    offset_ms: Annotated[str, Form()] = "",
):
    verify_api_key(request.headers.get("authorization"))

    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio chunk")
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio chunk exceeds 5 MB")

    suffix = Path(audio.filename or "live-audio.webm").suffix or ".webm"
    temp = AUDIO_DIR / f"live-{uuid.uuid4()}{suffix}"
    temp.write_bytes(data)

    try:
        result = transcribe_audio(temp)
        return {
            "meeting_id": meeting_id,
            "participant_id": participant_id,
            "speaker_name": speaker_name,
            "sequence": sequence,
            "offset_ms": int(offset_ms) if offset_ms else None,
            "text": result["text"],
            "segments": result["segments"],
        }
    finally:
        temp.unlink(missing_ok=True)


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
    except Exception:
        raise HTTPException(status_code=400, detail="metadata must be a JSON array")

    metadata_by_filename = {
        item["filename"]: item for item in metadata_items
        if isinstance(item, dict) and "filename" in item and "participant_id" in item
    }
    job_id = str(uuid.uuid4())
    job_audio_dir = AUDIO_DIR / job_id
    job_output_dir = OUTPUT_DIR / job_id
    job_audio_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)
    tracks = []

    for upload in files:
        original_filename = upload.filename or f"{uuid.uuid4()}.mkv"
        item = metadata_by_filename.get(original_filename)
        if not item:
            raise HTTPException(status_code=400, detail=f"Missing participant metadata for {original_filename}")
        destination = job_audio_dir / Path(original_filename).name
        with destination.open("wb") as output:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        await upload.close()
        tracks.append({
            "path": str(destination),
            "filename": original_filename,
            "participant_id": item["participant_id"],
            "speaker_name": item.get("speaker_name"),
        })

    job = {
        "job_id": job_id,
        "meeting_id": meeting_id,
        "recording_id": recording_id,
        "created_at": None,
        "status": "queued",
        "tracks": tracks,
        "output_dir": str(job_output_dir),
        "transcript_path": None,
        "text_transcript_path": None,
        "error": None,
    }
    path = JOBS_DIR / f"{job_id}.json"
    path.write_text(json.dumps(job, indent=2), encoding="utf-8")
    return {"job_id": job_id, "status": "queued", "meeting_id": meeting_id, "recording_id": recording_id}


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
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Transcription is not completed")
    path = Path(job["transcript_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript file missing")
    return FileResponse(path=path, media_type="application/json", filename="transcript.json")
