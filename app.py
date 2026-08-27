import json
import asyncio
import logging
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from config import (
    AI_API_KEY,
    AUDIO_DIR,
    JOBS_DIR,
    OUTPUT_DIR,
    APP_NAME,
    MODEL_WARMUP,
    ensure_directories,
)
from summarizer import summarize_meeting
from worker import get_whisper_model, start_worker, transcribe_audio

logger = logging.getLogger("defcomm-ai")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=APP_NAME, version="1.3.0")


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

    # Never block application startup on multi-gigabyte model loading.
    # Railway's /health endpoint can respond immediately. Set MODEL_WARMUP=true
    # only when you explicitly want a background warmup after startup.
    if MODEL_WARMUP:
        import threading

        def warm_models() -> None:
            try:
                get_whisper_model()
                # Import is intentionally local so the summary model is also lazy.
                from summarizer import _load_model
                _load_model()
                logger.info("AI models warmed successfully")
            except Exception:
                logger.exception("AI model warmup failed; models remain lazy-loaded")

        threading.Thread(
            target=warm_models,
            name="defcomm-ai-model-warmup",
            daemon=True,
        ).start()

    logger.info("DefComm AI service started")


@app.get("/health")
def health():
    # Lightweight liveness check: do not load Whisper or the summary model here.
    from summarizer import SUMMARY_MODEL

    return {
        "ok": True,
        "service": APP_NAME,
        "summary_model": SUMMARY_MODEL,
        "whisper_loaded": __import__("worker")._model is not None,
    }




def verify_websocket_api_key(websocket: WebSocket) -> None:
    if not AI_API_KEY:
        return
    authorization = websocket.headers.get("authorization")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if authorization.removeprefix("Bearer ") != AI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid AI service API key")


@app.websocket("/v1/live/transcribe/ws")
async def live_transcribe_websocket(websocket: WebSocket):
    """Persistent SFU -> AI live transcription stream.

    Protocol:
      1. SFU connects with Authorization: Bearer <AI_API_KEY>.
      2. SFU sends JSON session.start.
      3. For each chunk SFU sends JSON {type: audio, sequence, offset_ms} followed
         by one binary WebM/Ogg/Opus payload.
      4. AI returns {type: transcript, sequence, result: {...}}.
      5. SFU sends JSON session.stop and closes the socket.
    """
    try:
        verify_websocket_api_key(websocket)
        await websocket.accept()

        first = await websocket.receive_json()
        if first.get("type") != "session.start":
            await websocket.send_json({
                "type": "error",
                "message": "first message must be session.start",
            })
            await websocket.close(code=1008)
            return

        meeting_id = str(first.get("meeting_id", ""))
        participant_id = str(first.get("participant_id", ""))
        speaker_name = str(first.get("speaker_name") or "Participant")

        if not meeting_id or not participant_id:
            await websocket.send_json({
                "type": "error",
                "message": "meeting_id and participant_id are required",
            })
            await websocket.close(code=1008)
            return

        await websocket.send_json({
            "type": "session.ready",
            "meeting_id": meeting_id,
            "participant_id": participant_id,
        })

        logger.info(
            "Live transcription WebSocket connected meeting=%s participant=%s speaker=%s",
            meeting_id,
            participant_id,
            speaker_name,
        )

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect

            text = message.get("text")
            if text is not None:
                try:
                    control = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "invalid JSON control message",
                    })
                    continue

                message_type = control.get("type")
                if message_type == "session.stop":
                    await websocket.send_json({"type": "session.stopped"})
                    break
                if message_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue
                if message_type != "audio":
                    await websocket.send_json({
                        "type": "error",
                        "message": f"unsupported message type: {message_type}",
                    })
                    continue

                sequence = int(control.get("sequence", 0))
                offset_ms = int(control.get("offset_ms", 0))
                audio_message = await websocket.receive()
                binary = audio_message.get("bytes")
                if not binary:
                    await websocket.send_json({
                        "type": "error",
                        "sequence": sequence,
                        "message": "audio binary frame missing after audio metadata",
                    })
                    continue

                suffix = ".ogg" if binary.startswith(b"OggS") else ".webm"
                temp = AUDIO_DIR / f"live-ws-{uuid.uuid4()}{suffix}"
                temp.write_bytes(binary)
                try:
                    # Whisper is CPU-heavy and protected by a process-global lock.
                    # Run it off the ASGI event loop so one transcription cannot
                    # stall WebSocket heartbeats or other sessions.
                    result = await run_in_threadpool(transcribe_audio, temp)
                finally:
                    temp.unlink(missing_ok=True)

                await websocket.send_json({
                    "type": "transcript",
                    "sequence": sequence,
                    "offset_ms": offset_ms,
                    "meeting_id": meeting_id,
                    "participant_id": participant_id,
                    "speaker_name": speaker_name,
                    "result": result,
                })
                continue

            if message.get("bytes") is not None:
                await websocket.send_json({
                    "type": "error",
                    "message": "binary audio requires a preceding audio metadata message",
                })

    except WebSocketDisconnect:
        logger.info(
            "Live transcription WebSocket disconnected meeting=%s participant=%s",
            locals().get("meeting_id", ""),
            locals().get("participant_id", ""),
        )
    except HTTPException as error:
        # Authentication errors happen before accept; return a normal HTTP-style
        # close when possible. FastAPI/Starlette may surface this during handshake.
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1008, reason=str(error.detail))
    except Exception:
        logger.exception(
            "Live transcription WebSocket failed meeting=%s participant=%s",
            locals().get("meeting_id", ""),
            locals().get("participant_id", ""),
        )
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({
                    "type": "error",
                    "message": "live transcription failed",
                })
                await websocket.close(code=1011)
        except Exception:
            pass

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
        item["filename"]: item
        for item in metadata_items
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
            raise HTTPException(
                status_code=400,
                detail=f"Missing participant metadata for {original_filename}",
            )

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
    return {
        "job_id": job_id,
        "status": "queued",
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
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Transcription is not completed")

    path = Path(job["transcript_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Transcript file missing")

    return FileResponse(path=path, media_type="application/json", filename="transcript.json")


class ImportantMoment(BaseModel):
    timestamp_ms: int | None = None
    title: str | None = None
    description: str | None = None


class SummarySegment(BaseModel):
    speaker_id: str | None = None
    speaker_name: str | None = None
    sequence: int | None = None
    text: str
    start_ms: int | None = None
    end_ms: int | None = None


class MeetingSummaryRequest(BaseModel):
    meeting_id: str
    session_id: str | None = None
    segments: list[SummarySegment] = Field(default_factory=list)


@app.post("/v1/meeting/summarize")
def summarize_meeting_endpoint(
    request: Request,
    payload: MeetingSummaryRequest,
):
    verify_api_key(request.headers.get("authorization"))

    segments: list[dict[str, Any]] = [item.model_dump() for item in payload.segments]
    logger.info(
        "Generating local meeting summary meeting=%s session=%s segments=%s",
        payload.meeting_id,
        payload.session_id,
        len(segments),
    )

    try:
        summary = summarize_meeting(segments)
    except Exception as error:
        logger.exception("Local meeting summary generation failed")
        raise HTTPException(status_code=500, detail=f"summary generation failed: {error}") from error

    return {
        "meeting_id": payload.meeting_id,
        "session_id": payload.session_id,
        "summary": summary,
    }
