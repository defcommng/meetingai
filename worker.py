
import os

os.environ.setdefault(
    "HF_HOME",
    "/tmp/huggingface",
)

os.environ.setdefault(
    "HF_HUB_CACHE",
    "/tmp/huggingface/hub",
)


import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from config import (
    JOBS_DIR,
    MODEL_DIR,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
    WHISPER_VAD_FILTER,
    WORKER_POLL_SECONDS,
)

logger = logging.getLogger("defcomm-ai-worker")


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
    download_root=str(MODEL_DIR),
)

logger.info("Whisper model loaded")


def recover_interrupted_jobs() -> None:
    for path in JOBS_DIR.glob("*.json"):
        try:
            job = read_json(path)
        except Exception:
            continue

        if job.get("status") == "processing":
            job["status"] = "queued"
            job["error"] = None
            job["recovered_at"] = utc_now()
            write_json(path, job)
            logger.warning("Recovered interrupted job=%s", job.get("job_id"))


def transcribe_audio(audio_path: Path) -> dict[str, Any]:
    logger.info("Transcribing audio=%s", audio_path)

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=WHISPER_VAD_FILTER,
    )

    transcript_segments = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        transcript_segments.append(
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
        "segments": transcript_segments,
    }


def process_job(job_path: Path) -> None:
    job = read_json(job_path)
    job_id = job["job_id"]

    job["status"] = "processing"
    job["started_at"] = utc_now()
    job["error"] = None
    write_json(job_path, job)

    try:
        all_segments: list[dict[str, Any]] = []
        languages: list[str] = []

        for track in job["tracks"]:
            audio_path = Path(track["path"])
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

            result = transcribe_audio(audio_path)
            if result.get("language"):
                languages.append(result["language"])

            for segment in result["segments"]:
                all_segments.append(
                    {
                        "start": segment["start"],
                        "end": segment["end"],
                        "speaker_id": track["participant_id"],
                        "speaker_name": track.get("speaker_name"),
                        "text": segment["text"],
                    }
                )

        all_segments.sort(key=lambda item: (item["start"], item["end"]))

        output_dir = Path(job["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        transcript = {
            "version": 1,
            "job_id": job_id,
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

        job["status"] = "completed"
        job["completed_at"] = utc_now()
        job["transcript_path"] = str(transcript_path)
        job["text_transcript_path"] = str(text_path)
        job["segment_count"] = len(all_segments)
        job["error"] = None
        write_json(job_path, job)

        logger.info("Completed job=%s segments=%d", job_id, len(all_segments))

    except Exception as error:
        logger.exception("Job failed=%s", job_id)
        job["status"] = "failed"
        job["completed_at"] = utc_now()
        job["error"] = str(error)
        write_json(job_path, job)


def get_queued_jobs() -> list[Path]:
    jobs: list[Path] = []
    for path in JOBS_DIR.glob("*.json"):
        try:
            job = read_json(path)
        except Exception:
            continue
        if job.get("status") == "queued":
            jobs.append(path)
    jobs.sort(key=lambda path: path.stat().st_mtime)
    return jobs


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
    recover_interrupted_jobs()
    thread = threading.Thread(
        target=worker_loop,
        name="defcomm-transcription-worker",
        daemon=True,
    )
    thread.start()
    return thread
