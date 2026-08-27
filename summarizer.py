import json
import logging
import os
import re
import threading
from typing import Any

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger("defcomm-ai-summarizer")

SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "google/flan-t5-small")
SUMMARY_MAX_INPUT_TOKENS = int(os.getenv("SUMMARY_MAX_INPUT_TOKENS", "1800"))
SUMMARY_MAX_NEW_TOKENS = int(os.getenv("SUMMARY_MAX_NEW_TOKENS", "384"))
SUMMARY_NUM_BEAMS = int(os.getenv("SUMMARY_NUM_BEAMS", "2"))
SUMMARY_CHUNK_CHARS = int(os.getenv("SUMMARY_CHUNK_CHARS", "7000"))
SUMMARY_MODEL_CACHE = os.getenv("SUMMARY_MODEL_CACHE", "/tmp/defcomm-ai/models")

_model = None
_tokenizer = None
_model_lock = threading.Lock()
_load_lock = threading.Lock()


def _load_model() -> tuple[Any, Any]:
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    with _load_lock:
        if _model is not None and _tokenizer is not None:
            return _model, _tokenizer

        logger.info("Loading local summary model=%s", SUMMARY_MODEL)
        _tokenizer = AutoTokenizer.from_pretrained(
            SUMMARY_MODEL,
            cache_dir=SUMMARY_MODEL_CACHE,
        )
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            SUMMARY_MODEL,
            cache_dir=SUMMARY_MODEL_CACHE,
        )
        _model.eval()
        logger.info("Local summary model loaded successfully")

    return _model, _tokenizer


def _clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    return value


def format_transcript(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for segment in segments:
        text = _clean_text(str(segment.get("text", "")))
        if not text:
            continue
        speaker = _clean_text(str(segment.get("speaker_name") or segment.get("speaker_id") or "Speaker"))
        start_ms = segment.get("start_ms")
        if isinstance(start_ms, (int, float)):
            timestamp = f"{max(0, int(start_ms) // 1000)}s"
        else:
            timestamp = ""
        prefix = f"[{timestamp}] " if timestamp else ""
        lines.append(f"{prefix}{speaker}: {text}")
    return "\n".join(lines)


def _chunks(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in text.splitlines():
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            pieces.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len

    if current:
        pieces.append("\n".join(current))
    return pieces


def _generate(prompt: str) -> str:
    model, tokenizer = _load_model()
    with _model_lock:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=SUMMARY_MAX_INPUT_TOKENS,
        )
        outputs = model.generate(
            **inputs,
            max_new_tokens=SUMMARY_MAX_NEW_TOKENS,
            num_beams=max(1, SUMMARY_NUM_BEAMS),
            do_sample=False,
            early_stopping=True,
        )
        return _clean_text(tokenizer.decode(outputs[0], skip_special_tokens=True))


def _fallback_summary(segments: list[dict[str, Any]], reason: str | None = None) -> dict[str, Any]:
    usable = [s for s in segments if _clean_text(str(s.get("text", "")))]
    if not usable:
        overview = "No spoken content was captured for this meeting."
    else:
        first = _clean_text(str(usable[0].get("text", "")))
        overview = f"The meeting transcript contains {len(usable)} spoken segments. It begins with: {first}"

    result = {
        "overview": overview,
        "topics": [],
        "decisions": [],
        "action_items": [],
        "important_moments": [],
    }
    if reason:
        result["overview"] += f" Local AI summarization was unavailable ({reason})."
    return result


def _parse_json_output(raw: str) -> dict[str, Any] | None:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
        candidate = re.sub(r"```$", "", candidate).strip()

    try:
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            return None
    return None


def _normalize_summary(value: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    def strings(key: str) -> list[str]:
        data = value.get(key, [])
        if isinstance(data, str):
            data = [data]
        if not isinstance(data, list):
            return []
        return [_clean_text(str(item)) for item in data if _clean_text(str(item))]

    moments = value.get("important_moments", [])
    if not isinstance(moments, list):
        moments = []

    normalized_moments: list[dict[str, Any]] = []
    for item in moments:
        if isinstance(item, str):
            normalized_moments.append({"timestamp_ms": None, "title": item, "description": None})
            continue
        if not isinstance(item, dict):
            continue
        normalized_moments.append({
            "timestamp_ms": item.get("timestamp_ms"),
            "title": _clean_text(str(item.get("title") or "")) or None,
            "description": _clean_text(str(item.get("description") or "")) or None,
        })

    overview = _clean_text(str(value.get("overview") or value.get("text") or ""))
    if not overview:
        overview = _fallback_summary(segments)["overview"]

    return {
        "overview": overview,
        "topics": strings("topics"),
        "decisions": strings("decisions"),
        "action_items": strings("action_items"),
        "important_moments": normalized_moments,
    }


def summarize_meeting(segments: list[dict[str, Any]]) -> dict[str, Any]:
    transcript = format_transcript(segments)
    if not transcript:
        return _fallback_summary(segments)

    chunks = _chunks(transcript, SUMMARY_CHUNK_CHARS)
    partials: list[str] = []

    for index, chunk in enumerate(chunks):
        prompt = (
            "Summarize this meeting transcript chunk. Be factual and concise. "
            "Return ONLY JSON with this exact shape: "
            '{"overview":"...","topics":[],"decisions":[],"action_items":[],"important_moments":[]}\n\n'
            f"Transcript chunk {index + 1} of {len(chunks)}:\n{chunk}"
        )
        raw = _generate(prompt)
        parsed = _parse_json_output(raw)
        if parsed:
            partials.append(json.dumps(parsed, ensure_ascii=False))
        else:
            partials.append(raw)

    if len(partials) == 1:
        parsed = _parse_json_output(partials[0])
        if parsed:
            return _normalize_summary(parsed, segments)

    merge_prompt = (
        "Combine the following meeting-summary drafts into one accurate final summary. "
        "Remove duplicates and do not invent facts. Return ONLY valid JSON with this exact shape: "
        '{"overview":"...","topics":[],"decisions":[],"action_items":[],"important_moments":[]}\n\n'
        "Drafts:\n" + "\n\n---\n\n".join(partials)
    )
    raw = _generate(merge_prompt)
    parsed = _parse_json_output(raw)
    if parsed:
        return _normalize_summary(parsed, segments)

    return _fallback_summary(segments, reason="model returned non-JSON output")
