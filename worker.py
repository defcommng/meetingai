import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", "/tmp/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.getenv("HF_HUB_CACHE", "/tmp/huggingface/hub"))

from faster_whisper import WhisperModel

from config import (
    JOBS_DIR,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
    WHISPER_VAD_FILTER,
    WORKER_POLL_SECONDS,
)

logger = logging.getLogger("defcomm-ai-worker")

logger.info(
    "Loading Whisper model=%s device=%s compute_type=%s",
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
)

model = WhisperModel(
    WHISPER_MODEL,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE,
    download_root=os.getenv("WHISPER_MODEL_CACHE", "/tmp/defcomm-ai/models"),
)

model_lock = threading.Lock()
logger.info("Whisper model loaded successfully")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    temporary.replace(path)


def transcribe_audio(path: Path) -> dict[str, Any]:
    with model_lock:
        segments, info = model.transcribe(
            str(path),
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=WHISPER_VAD_FILTER,
            condition_on_previous_text=False,
        )

        items = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                items.append(
                    {
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": text,
                    }
                )

    return {
        "language": info.language,
        "language_probability": float(info.language_probability),
        "duration": float(info.duration),
        "segments": items,
        "text": " ".join(item["text"] for item in items).strip(),
    }


def process_job(job_path: Path) -> None:
    job = read_json(job_path)
    job["status"] = "processing"
    job["started_at"] = utc_now()
    job["error"] = None
    write_json(job_path, job)

    try:
        all_segments = []
        languages = []

        for track in job["tracks"]:
            result = transcribe_audio(Path(track["path"]))
            if result.get("language"):
                languages.append(result["language"])
            for segment in result["segments"]:
                all_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "speaker_id": track["participant_id"],
                    "speaker_name": track.get("speaker_name"),
                    "text": segment["text"],
                })

        all_segments.sort(key=lambda item: (item["start"], item["end"]))
        output_dir = Path(job["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        transcript = {
            "version": 1,
            "job_id": job["job_id"],
            "meeting_id": job["meeting_id"],
            "recording_id": job["recording_id"],
            "created_at": utc_now(),
            "language": max(set(languages), key=languages.count) if languages else None,
            "segments": all_segments,
        }
        transcript_path = output_dir / "transcript.json"
        write_json(transcript_path, transcript)

        text_path = output_dir / "transcript.txt"
        with text_path.open("w", encoding="utf-8") as file:
            for segment in all_segments:
                speaker = segment.get("speaker_name") or segment["speaker_id"]
                file.write(
                    f"[{segment['start']:.2f} - {segment['end']:.2f}] "
                    f"{speaker}: {segment['text']}\n"
                )

        job.update({
            "status": "completed",
            "completed_at": utc_now(),
            "transcript_path": str(transcript_path),
            "text_transcript_path": str(text_path),
            "segment_count": len(all_segments),
            "error": None,
        })
        write_json(job_path, job)
        logger.info("Job %s completed: %s segments", job["job_id"], len(all_segments))
    except Exception as error:
        logger.exception("Job %s failed", job["job_id"])
        job.update({"status": "failed", "completed_at": utc_now(), "error": str(error)})
        write_json(job_path, job)


def get_queued_jobs() -> list[Path]:
    jobs = []
    for path in JOBS_DIR.glob("*.json"):
        try:
            if read_json(path).get("status") == "queued":
                jobs.append(path)
        except Exception:
            continue
    return sorted(jobs, key=lambda path: path.stat().st_mtime)


def worker_loop() -> None:
    logger.info("Worker loop started")
    while True:
        try:
            jobs = get_queued_jobs()
            if jobs:
                process_job(jobs[0])
            else:
                time.sleep(WORKER_POLL_SECONDS)
        except Exception:
            logger.exception("Unexpected worker error")
            time.sleep(WORKER_POLL_SECONDS)


def start_worker() -> threading.Thread:
    thread = threading.Thread(target=worker_loop, name="defcomm-transcription-worker", daemon=True)
    thread.start()
    return thread
