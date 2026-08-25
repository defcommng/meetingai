# DefComm AI transcription service

This service runs faster-whisper locally and accepts participant-level microphone recordings from the DefComm SFU.

## Render

Create a Python Web Service from this directory. Attach a persistent disk at `/var/data`.

Required environment variables:

- `AI_API_KEY` - long random secret shared with the SFU.
- `WHISPER_MODEL=small`
- `WHISPER_DEVICE=cpu`
- `WHISPER_COMPUTE_TYPE=int8`
- `DATA_DIR=/var/data`

The first startup downloads the Whisper model into `/var/data/models` so the persistent disk can retain it across restarts.

## Endpoints

- `GET /health`
- `POST /v1/transcriptions`
- `GET /v1/transcriptions/{job_id}`
- `GET /v1/transcriptions/{job_id}/transcript`
- `GET /v1/transcriptions/{job_id}/text`

The SFU submits each participant microphone `.mkv` as a separate file with metadata containing `participant_id` and `speaker_name`.
