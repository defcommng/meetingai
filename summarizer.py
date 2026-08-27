import gc
import json
import logging
import os
import re
import threading
from typing import Any

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import (
    SUMMARY_CHUNK_CHARS,
    SUMMARY_MAX_INPUT_TOKENS,
    SUMMARY_MAX_NEW_TOKENS,
    SUMMARY_MODEL,
    SUMMARY_MODEL_CACHE,
    SUMMARY_NUM_BEAMS,
)

logger = logging.getLogger("defcomm-ai-summarizer")

_model: Any = None
_tokenizer: Any = None
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


def unload_model() -> None:
    """Release the local summary model when RAM pressure matters."""
    global _model, _tokenizer
    with _load_lock:
        _model = None
        _tokenizer = None
        gc.collect()
        logger.info("Local summary model unloaded")


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _segment_time_ms(segment: dict[str, Any]) -> int | None:
    value = segment.get("start_ms")
    if isinstance(value, (int, float)):
        return max(0, int(value))

    value = segment.get("start")
    if isinstance(value, (int, float)):
        return max(0, int(float(value) * 1000))
    return None


def format_transcript(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for segment in segments:
        text = _clean_text(segment.get("text"))
        if not text:
            continue

        speaker = _clean_text(
            segment.get("speaker_name")
            or segment.get("speaker_id")
            or "Speaker"
        )
        start_ms = _segment_time_ms(segment)
        timestamp = f"[{start_ms // 1000}s] " if start_ms is not None else ""
        lines.append(f"{timestamp}{speaker}: {text}")

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
        )

        return _clean_text(
            tokenizer.decode(outputs[0], skip_special_tokens=True)
        )


def _parse_json_output(raw: str) -> dict[str, Any] | None:
    candidate = raw.strip()
    if not candidate:
        return None

    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)

    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if not match:
        return None

    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []

    result: list[str] = []
    for item in value:
        text = _clean_text(item)
        if text:
            result.append(text)
    return result


def _normalize_summary(value: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    overview = _clean_text(value.get("overview") or value.get("summary") or value.get("text"))
    topics = _string_list(value.get("topics"))
    decisions = _string_list(value.get("decisions"))
    action_items = _string_list(value.get("action_items"))

    moments_value = value.get("important_moments", [])
    if not isinstance(moments_value, list):
        moments_value = []

    moments: list[dict[str, Any]] = []
    for item in moments_value:
        if isinstance(item, str):
            title = _clean_text(item)
            if title:
                moments.append({
                    "timestamp_ms": None,
                    "title": title,
                    "description": None,
                })
        elif isinstance(item, dict):
            moments.append({
                "timestamp_ms": item.get("timestamp_ms"),
                "title": _clean_text(item.get("title")) or None,
                "description": _clean_text(item.get("description")) or None,
            })

    if not overview:
        overview = _fallback_summary(segments)["overview"]

    return {
        "overview": overview,
        "topics": topics,
        "decisions": decisions,
        "action_items": action_items,
        "important_moments": moments,
    }


def _fallback_summary(segments: list[dict[str, Any]], reason: str | None = None) -> dict[str, Any]:
    usable = [s for s in segments if _clean_text(s.get("text"))]
    speakers: list[str] = []
    seen: set[str] = set()

    for segment in usable:
        speaker = _clean_text(
            segment.get("speaker_name") or segment.get("speaker_id") or "Speaker"
        )
        if speaker and speaker not in seen:
            seen.add(speaker)
            speakers.append(speaker)

    if not usable:
        overview = "No spoken content was captured for this meeting."
    else:
        overview = (
            f"The meeting contains {len(usable)} spoken segments from "
            f"{len(speakers)} speaker(s)."
        )

    if reason:
        overview += f" AI summarization was unavailable ({reason})."

    return {
        "overview": overview,
        "topics": [],
        "decisions": [],
        "action_items": [],
        "important_moments": [],
    }


def summarize_meeting(segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a post-meeting summary from the final transcript."""
    transcript = format_transcript(segments)
    if not transcript:
        return _fallback_summary(segments)

    chunks = _chunks(transcript, SUMMARY_CHUNK_CHARS)
    drafts: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks):
        prompt = (
            "You are summarizing a business meeting transcript. "
            "Use only information present in the transcript. Do not invent names, "
            "decisions, dates, or action items. Return ONLY valid JSON. "
            "The JSON must have exactly these keys: overview, topics, decisions, "
            "action_items, important_moments. topics, decisions and action_items "
            "must be arrays of strings. important_moments must be an array of "
            "objects with timestamp_ms, title, description.\n\n"
            f"Transcript chunk {index + 1}/{len(chunks)}:\n{chunk}"
        )

        try:
            raw = _generate(prompt)
            parsed = _parse_json_output(raw)
            if parsed:
                drafts.append(parsed)
            elif raw:
                drafts.append({"overview": raw})
        except Exception:
            logger.exception("Summary chunk %s/%s failed", index + 1, len(chunks))

    if not drafts:
        return _fallback_summary(segments, reason="summary model produced no usable output")

    if len(drafts) == 1:
        return _normalize_summary(drafts[0], segments)

    merge_source = json.dumps(drafts, ensure_ascii=False)
    merge_prompt = (
        "Combine these partial meeting summaries into one final summary. "
        "Remove duplicates. Never invent facts. Return ONLY valid JSON with "
        "exactly these keys: overview, topics, decisions, action_items, "
        "important_moments.\n\n"
        f"Partial summaries:\n{merge_source}"
    )

    try:
        raw = _generate(merge_prompt)
        parsed = _parse_json_output(raw)
        if parsed:
            return _normalize_summary(parsed, segments)
    except Exception:
        logger.exception("Final summary merge failed")

    # A deterministic fallback is better than returning an empty summary.
    return _normalize_summary(drafts[0], segments)
