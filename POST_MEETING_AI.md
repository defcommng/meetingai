# Post-meeting batch transcription changes

`POST /v1/transcriptions` is the production transcription path. Multiple microphone recordings can be
uploaded in one request with per-track metadata including `offset_ms`, `participant_id`, and `speaker_name`.

The worker adds each track's offset to Whisper's track-local timestamps before merging all speakers into
a single meeting timeline.

The live `/v1/live/transcribe` HTTP endpoint and `/v1/live/transcribe/ws` WebSocket may remain for backwards
compatibility, but the SFU no longer depends on either for meeting transcription.
