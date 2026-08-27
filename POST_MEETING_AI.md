# DefComm AI — post-meeting transcription and summary

The AI service is intentionally **post-meeting**. There is no live transcription path and no WebSocket dependency.

## Flow

1. SFU uploads the completed microphone recordings to `POST /v1/transcriptions`.
2. The AI worker queues the job and immediately returns a `job_id`.
3. The worker transcribes every uploaded track with Whisper and merges the track-local timestamps using `offset_ms`.
4. The worker writes `transcript.json` and `transcript.txt`.
5. The worker generates a local summary and writes `summary.json`.
6. SFU can poll `GET /v1/transcriptions/{job_id}` or fetch the result files directly.

## Endpoints

- `GET /health`
- `POST /v1/transcriptions`
- `GET /v1/transcriptions/{job_id}`
- `GET /v1/transcriptions/{job_id}/transcript`
- `GET /v1/transcriptions/{job_id}/summary`
- `POST /v1/transcriptions/{job_id}/retry`

The old `/v1/live/transcribe` endpoint is intentionally removed.
