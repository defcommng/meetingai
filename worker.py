import gc
import json
import logging
import multiprocessing
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
    SUMMARY_ENABLED,
    UNLOAD_SUMMARY_AFTER_JOB,
    UNLOAD_WHISPER_BEFORE_SUMMARY,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
    WHISPER_MODEL_CACHE,
    WHISPER_VAD_FILTER,
    WORKER_POLL_SECONDS,
    WORKER_STALE_SECONDS,
    MODEL_WARMUP,
)
from summarizer import summarize_meeting, unload_model as unload_summary_model, _load_model

logger = logging.getLogger("defcomm-ai-worker")

_model: WhisperModel | None = None
_model_lock = threading.Lock()
_model_load_lock = threading.Lock()

_worker_process: multiprocessing.Process | None = None
_worker_supervisor: threading.Thread | None = None


def get_whisper_model() -> WhisperModel:
    global _model

    if _model is not None:
        return _model

    with _model_load_lock:
        if _model is not None:
            return _model

        logger.info(
            "Loading Whisper model=%s device=%s compute_type=%s",
            WHISPER_MODEL,
            WHISPER_DEVICE,
            WHISPER_COMPUTE_TYPE,
        )

        _model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
            download_root=WHISPER_MODEL_CACHE,
        )
        logger.info("Whisper model loaded successfully")
        return _model


def unload_whisper_model() -> None:
    global _model
    with _model_load_lock:
        _model = None
        gc.collect()
        logger.info("Whisper model unloaded")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    temporary.replace(path)


def transcribe_audio(path: Path) -> dict[str, Any]:
    model = get_whisper_model()

    with _model_lock:
        segments, info = model.transcribe(
            str(path),
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=WHISPER_VAD_FILTER,
            condition_on_previous_text=False,
        )

        items: list[dict[str, Any]] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            items.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
            })

    return {
        "language": info.language,
        "language_probability": float(info.language_probability),
        "duration": float(info.duration),
        "segments": items,
        "text": " ".join(item["text"] for item in items).strip(),
    }


def _recover_abandoned_jobs() -> None:
    """Requeue jobs that were left in processing when the worker process died/restarted."""
    for path in JOBS_DIR.glob("*.json"):
        try:
            job = read_json(path)
            if job.get("status") != "processing":
                continue
            job["status"] = "queued"
            job["stage"] = "queued"
            job["error"] = "Recovered processing job after worker restart"
            job["recovered_at"] = utc_now()
            job["started_at"] = None
            write_json(path, job)
            logger.warning("Recovered abandoned processing job %s", job.get("job_id"))
        except Exception:
            logger.exception("Failed recovering abandoned job %s", path)


def _recover_stale_jobs() -> None:
    now = time.time()
    cutoff = now - WORKER_STALE_SECONDS

    for path in JOBS_DIR.glob("*.json"):
        try:
            job = read_json(path)
            if job.get("status") != "processing":
                continue

            started_at = job.get("started_at")
            if not started_at:
                continue

            parsed = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            if parsed.timestamp() >= cutoff:
                continue

            job["status"] = "queued"
            job["stage"] = "queued"
            job["error"] = "Recovered stale processing job after worker timeout"
            job["recovered_at"] = utc_now()
            job["started_at"] = None
            write_json(path, job)
            logger.warning("Recovered stale job %s", job.get("job_id"))
        except Exception:
            logger.exception("Failed checking stale job %s", path)


def _write_transcript(job: dict[str, Any], all_segments: list[dict[str, Any]], languages: list[str]) -> Path:
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript = {
        "version": 2,
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

    return transcript_path


def _write_summary(job: dict[str, Any], summary: dict[str, Any]) -> Path:
    output_dir = Path(job["output_dir"])
    summary_path = output_dir / "summary.json"
    write_json(summary_path, {
        "version": 1,
        "job_id": job["job_id"],
        "meeting_id": job["meeting_id"],
        "recording_id": job["recording_id"],
        "created_at": utc_now(),
        "summary": summary,
    })
    return summary_path


def process_job(job_path: Path) -> None:
    job = read_json(job_path)
    job["status"] = "processing"
    job["started_at"] = utc_now()
    job["updated_at"] = utc_now()
    job["error"] = None
    job["stage"] = "transcribing"
    job["current_track"] = 0
    job["total_tracks"] = len(job.get("tracks", []))
    write_json(job_path, job)

    try:
        all_segments: list[dict[str, Any]] = []
        languages: list[str] = []

        for index, track in enumerate(job.get("tracks", []), start=1):
            job["current_track"] = index
            job["stage"] = "transcribing"
            job["updated_at"] = utc_now()
            write_json(job_path, job)

            logger.info(
                "Job %s transcribing track %s/%s participant=%s",
                job["job_id"],
                index,
                len(job.get("tracks", [])),
                track.get("participant_id"),
            )

            result = transcribe_audio(Path(track["path"]))
            language = result.get("language")
            if language:
                languages.append(language)

            offset_seconds = max(
                0.0,
                float(track.get("offset_ms") or 0) / 1000.0,
            )

            for segment in result.get("segments", []):
                all_segments.append({
                    "start": offset_seconds + float(segment["start"]),
                    "end": offset_seconds + float(segment["end"]),
                    "speaker_id": track["participant_id"],
                    "speaker_name": track.get("speaker_name"),
                    "text": segment["text"],
                })

        all_segments.sort(key=lambda item: (item["start"], item["end"]))
        transcript_path = _write_transcript(job, all_segments, languages)

        job.update({
            "stage": "summarizing" if SUMMARY_ENABLED else "completed",
            "updated_at": utc_now(),
            "transcript_path": str(transcript_path),
            "text_transcript_path": str(Path(job["output_dir"]) / "transcript.txt"),
            "segment_count": len(all_segments),
        })
        write_json(job_path, job)

        if UNLOAD_WHISPER_BEFORE_SUMMARY:
            unload_whisper_model()

        summary_path: Path | None = None
        if SUMMARY_ENABLED:
            logger.info(
                "Generating local meeting summary meeting=%s recording=%s segments=%s",
                job["meeting_id"],
                job["recording_id"],
                len(all_segments),
            )
            summary = summarize_meeting(all_segments)
            summary_path = _write_summary(job, summary)

        job.update({
            "status": "completed",
            "stage": "completed",
            "updated_at": utc_now(),
            "completed_at": utc_now(),
            "summary_path": str(summary_path) if summary_path else None,
            "error": None,
        })
        write_json(job_path, job)

        logger.info(
            "Job %s completed: transcript_segments=%s summary=%s",
            job["job_id"],
            len(all_segments),
            bool(summary_path),
        )

        if UNLOAD_SUMMARY_AFTER_JOB:
            unload_summary_model()

    except Exception as error:
        logger.exception("Job %s failed", job.get("job_id"))
        job.update({
            "status": "failed",
            "stage": "failed",
            "completed_at": utc_now(),
            "updated_at": utc_now(),
            "error": str(error),
        })
        write_json(job_path, job)


def get_queued_jobs() -> list[Path]:
    jobs: list[Path] = []
    for path in JOBS_DIR.glob("*.json"):
        try:
            if read_json(path).get("status") == "queued":
                jobs.append(path)
        except Exception:
            continue
    return sorted(jobs, key=lambda path: path.stat().st_mtime)


def worker_loop() -> None:
    logger.info("Worker process loop started pid=%s", os.getpid())
    # Any processing job belongs to a previous worker process and is safe to resume.
    _recover_abandoned_jobs()

    if MODEL_WARMUP:
        try:
            get_whisper_model()
            if SUMMARY_ENABLED:
                _load_model()
            logger.info("AI models warmed successfully in worker process")
        except Exception:
            logger.exception("AI model warmup failed; lazy loading remains enabled")

    while True:
        try:
            _recover_stale_jobs()
            jobs = get_queued_jobs()
            if jobs:
                process_job(jobs[0])
            else:
                time.sleep(WORKER_POLL_SECONDS)
        except Exception:
            logger.exception("Unexpected worker error")
            time.sleep(WORKER_POLL_SECONDS)


def _spawn_worker() -> multiprocessing.Process:
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=worker_loop,
        name="defcomm-transcription-worker",
        daemon=True,
    )
    process.start()
    return process


def _supervise_worker() -> None:
    global _worker_process
    while True:
        process = _worker_process
        if process is not None:
            process.join()
            logger.error(
                "AI worker process exited exitcode=%s; restarting",
                process.exitcode,
            )
        time.sleep(3)
        try:
            _worker_process = _spawn_worker()
        except Exception:
            logger.exception("Failed to restart AI worker process")
            time.sleep(5)


def start_worker() -> multiprocessing.Process:
    global _worker_process, _worker_supervisor
    if _worker_process is not None and _worker_process.is_alive():
        return _worker_process

    _worker_process = _spawn_worker()
    if _worker_supervisor is None or not _worker_supervisor.is_alive():
        _worker_supervisor = threading.Thread(
            target=_supervise_worker,
            name="defcomm-ai-worker-supervisor",
            daemon=True,
        )
        _worker_supervisor.start()
    return _worker_process


def worker_is_alive() -> bool:
    return bool(_worker_process is not None and _worker_process.is_alive())

