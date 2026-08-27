import json
import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
import httpx

from config import (
    AI_API_KEY,
    LLM_API_URL,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
    AUDIO_DIR,
    JOBS_DIR,
    LIVE_JOBS_DIR,
    OUTPUT_DIR,
    APP_NAME,
    ensure_directories,
)
from worker import start_worker, transcribe_audio, read_json, write_json, utc_now

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


# Actual implementation using a Request keeps multipart and Authorization clea@app.post("/v1/live/transcribe", status_code=202)
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

    job_id = uuid.uuid4()
    suffix = Path(audio.filename or "live-audio.ogg").suffix or ".ogg"
    audio_path = AUDIO_DIR / f"live-{job_id}{suffix}"
    audio_path.write_bytes(data)
    job_path = LIVE_JOBS_DIR / f"{job_id}.json"
    write_json(job_path, {
        "job_id": str(job_id),
        "meeting_id": meeting_id,
        "participant_id": participant_id,
        "speaker_name": speaker_name,
        "sequence": sequence,
        "offset_ms": int(offset_ms) if offset_ms else None,
        "audio_path": str(audio_path),
        "status": "queued",
        "created_at": utc_now(),
        "text": "",
        "segments": [],
        "error": None,
    })
    return {"job_id": str(job_id), "status": "queued"}


@app.get("/v1/live/transcribe/{job_id}")
def get_live_transcription(job_id: str, request: Request):
    verify_api_key(request.headers.get("authorization"))
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID")
    path = LIVE_JOBS_DIR / f"{job_uuid}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Live transcription job not found")
    return read_json(path)
temp.unlink(missing_ok=True)





def _extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("AI summary did not contain a JSON object")
    return json.loads(cleaned[start : end + 1])


@app.post("/v1/meeting/summarize")
async def summarize_meeting(request: Request):
    verify_api_key(request.headers.get("authorization"))
    body = await request.json()
    meeting_id = body.get("meeting_id")
    segments = body.get("segments") or []
    if not meeting_id:
        raise HTTPException(status_code=400, detail="meeting_id is required")
    if not isinstance(segments, list):
        raise HTTPException(status_code=400, detail="segments must be an array")

    if not segments:
        return {
            "summary": {
                "overview": "No spoken content was captured for this meeting.",
                "topics": [],
                "decisions": [],
                "action_items": [],
                "important_moments": [],
            }
        }

    if not LLM_API_URL or not LLM_API_KEY or not LLM_MODEL:
        raise HTTPException(status_code=503, detail="Summary model is not configured")

    transcript_text = "\n".join(
        f"[{segment.get('start_ms') or 0}ms] "
        f"{segment.get('speaker_name') or segment.get('speaker_id') or 'Speaker'}: "
        f"{str(segment.get('text') or '').strip()}"
        for segment in segments
        if str(segment.get("text") or "").strip()
    )

    system_prompt = (
        "You are a meeting intelligence assistant. Summarize the supplied transcript "
        "faithfully. Do not invent facts, decisions, people, dates, or action items. "
        "Return ONLY valid JSON with this exact shape: "
        '{"overview":"string","topics":["string"],"decisions":["string"],'
        '"action_items":["string"],"important_moments":[{"timestamp_ms":0,"title":"string","description":"string"}]}'
    )
    user_prompt = (
        f"Meeting ID: {meeting_id}\n\nTranscript:\n{transcript_text}\n\n"
        "Produce the JSON summary now."
    )

    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
            response = await client.post(LLM_API_URL, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Summary provider request failed: {exc}") from exc

    if not response.is_success:
        raise HTTPException(status_code=502, detail=f"Summary provider returned {response.status_code}: {response.text[:1000]}")

    try:
        provider = response.json()
        content = provider["choices"][0]["message"]["content"]
        summary = _extract_json_object(content)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Invalid summary provider response: {exc}") from exc

    def list_of_strings(value):
        return [str(item).strip() for item in (value or []) if str(item).strip()]

    moments = []
    for item in summary.get("important_moments") or []:
        if isinstance(item, dict):
            moments.append({
                "timestamp_ms": item.get("timestamp_ms"),
                "title": item.get("title"),
                "description": item.get("description"),
            })

    return {
        "summary": {
            "overview": str(summary.get("overview") or "").strip() or None,
            "topics": list_of_strings(summary.get("topics")),
            "decisions": list_of_strings(summary.get("decisions")),
            "action_items": list_of_strings(summary.get("action_items")),
            "important_moments": moments,
        }
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
