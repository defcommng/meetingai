# DefComm AI Worker

Local speech-to-text and local meeting summarization service for DefComm.

## Models

- Whisper: `faster-whisper` for speech-to-text.
- Summary: `google/flan-t5-small` by default, loaded locally through Hugging Face Transformers.

No OpenAI/Anthropic/Gemini API key is required for meeting summaries.

## Required environment

```env
AI_API_KEY=replace-with-a-long-random-secret
WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
DATA_DIR=/var/data
WHISPER_MODEL_CACHE=/var/data/models
SUMMARY_MODEL_CACHE=/var/data/models
HF_HOME=/var/data/huggingface
HF_HUB_CACHE=/var/data/huggingface/hub
```

## Summary settings

```env
SUMMARY_MODEL=google/flan-t5-small
SUMMARY_MAX_INPUT_TOKENS=1800
SUMMARY_MAX_NEW_TOKENS=384
SUMMARY_NUM_BEAMS=2
SUMMARY_CHUNK_CHARS=7000
```

The summary route accepts the final speaker-attributed transcript from the SFU:

```text
POST /v1/meeting/summarize
Authorization: Bearer <AI_API_KEY>
Content-Type: application/json
```

Response:

```json
{
  "meeting_id": "...",
  "session_id": "...",
  "summary": {
    "overview": "...",
    "topics": [],
    "decisions": [],
    "action_items": [],
    "important_moments": []
  }
}
```

The model runs on the same service as Whisper. CPU-only inference can be slow for long meetings, so use persistent storage for model caches and give the Railway service enough RAM/CPU.
