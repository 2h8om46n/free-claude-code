"""Fold an Anthropic Messages SSE stream into a single JSON Message body.

Used when client requests omit ``stream`` or set ``stream: false``. The internal
pipeline is always SSE (provider execution always yields a stream), so callers
that need a non-streaming response consume the stream here and get back the same
shape the real Anthropic API returns for a non-streaming ``messages.create()``
call.
"""

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from free_claude_code.core.failures import ExecutionFailure, FailureKind

from .streaming.decoder import AnthropicSSEDecoder

__all__ = ["aggregate_anthropic_sse_to_message"]


def _sse_payload(raw: str) -> dict[str, Any] | None:
    data_parts: list[str] = []
    start = 0
    raw_len = len(raw)
    while start < raw_len:
        next_lf = raw.find("\n", start)
        next_cr = raw.find("\r", start)
        if next_lf == -1:
            end = next_cr
        elif next_cr == -1:
            end = next_lf
        else:
            end = next_cr if next_cr < next_lf else next_lf
        if end == -1:
            line = raw[start:]
            start = raw_len
        else:
            line = raw[start:end]
            start = end + 1
            if raw[end] == "\r" and start < raw_len and raw[start] == "\n":
                start += 1
        if line.startswith("data:"):
            data_parts.append(line[5:].strip())
    if not data_parts:
        return None
    data_text = "\n".join(data_parts)
    try:
        parsed = json.loads(data_text) if data_text else {}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"raw": data_text}


async def aggregate_anthropic_sse_to_message(
    stream: AsyncIterator[str],
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    """Assemble a complete Messages JSON body from an Anthropic SSE stream.

    Returns ``(message_body, error, complete)`` where ``error`` is the payload
    of a top-level ``event: error`` if one arrived and ``complete`` records
    whether the stream emitted ``message_stop``.
    """
    decoder = AnthropicSSEDecoder()
    message: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}
    parts: dict[int, list[str]] = {}
    error: dict[str, Any] | None = None
    complete = False

    def handle_payload(payload: dict[str, Any]) -> None:
        nonlocal message, error, complete
        ptype = payload.get("type")
        if ptype == "message_start":
            started = payload.get("message")
            if isinstance(started, dict):
                message = dict(started)
        elif ptype == "content_block_start":
            idx = payload.get("index")
            block = payload.get("content_block")
            if isinstance(idx, int) and isinstance(block, dict):
                blocks[idx] = dict(block)
                parts.setdefault(idx, [])
        elif ptype == "content_block_delta":
            idx = payload.get("index")
            delta = payload.get("delta")
            if not isinstance(idx, int) or not isinstance(delta, dict):
                return
            if idx not in blocks:
                blocks[idx] = {"type": "text", "text": ""}
                parts[idx] = []
            dtype = delta.get("type")
            if dtype == "text_delta":
                parts[idx].append(str(delta.get("text", "")))
            elif dtype == "thinking_delta":
                parts[idx].append(str(delta.get("thinking", "")))
            elif dtype == "input_json_delta":
                parts[idx].append(str(delta.get("partial_json", "")))
            elif dtype == "signature_delta":
                blocks[idx]["signature"] = str(blocks[idx].get("signature", "")) + str(
                    delta.get("signature", "")
                )
            elif dtype == "citations_delta":
                citation = delta.get("citation")
                citations = blocks[idx].get("citations")
                if citations is None:
                    citations = []
                if isinstance(citation, dict) and isinstance(citations, list):
                    blocks[idx]["citations"] = [*citations, dict(citation)]
        elif ptype == "message_delta":
            delta = payload.get("delta")
            if isinstance(delta, dict):
                if delta.get("stop_reason"):
                    message["stop_reason"] = delta["stop_reason"]
                if "stop_sequence" in delta:
                    message["stop_sequence"] = delta["stop_sequence"]
            usage = payload.get("usage")
            if isinstance(usage, dict):
                merged = (
                    dict(message["usage"])
                    if isinstance(message.get("usage"), dict)
                    else {}
                )
                merged.update(
                    {k: v for k, v in usage.items() if isinstance(v, int | dict)}
                )
                message["usage"] = merged
        elif ptype == "error":
            err = payload.get("error")
            error = (
                err
                if isinstance(err, dict)
                else {"type": "api_error", "message": "provider error"}
            )
        elif ptype == "message_stop":
            complete = True

    async for chunk in stream:
        for raw in decoder.feed_frames(chunk):
            payload = _sse_payload(raw)
            if payload is not None:
                handle_payload(payload)
    for raw in decoder.finish_frames():
        payload = _sse_payload(raw)
        if payload is not None:
            handle_payload(payload)

    content: list[dict[str, Any]] = []
    for idx in sorted(blocks):
        block = blocks[idx]
        accumulated = "".join(parts.get(idx, []))
        btype = block.get("type")
        if btype == "text":
            block["text"] = str(block.get("text", "")) + accumulated
        elif btype == "thinking":
            block["thinking"] = str(block.get("thinking", "")) + accumulated
            block.setdefault("signature", "")
        elif btype == "tool_use" and error is None:
            try:
                value = (
                    json.loads(accumulated) if parts.get(idx) else block.get("input")
                )
                if not isinstance(value, dict):
                    raise ValueError("Tool input must be an object.")
                json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise ExecutionFailure(
                    FailureKind.UPSTREAM,
                    502,
                    "Provider tool arguments cannot be represented as a Messages input object.",
                    False,
                ) from exc
            block["input"] = value
        content.append(block)

    message["content"] = content
    message.setdefault("id", f"msg_{uuid.uuid4()}")
    message.setdefault("type", "message")
    message.setdefault("role", "assistant")
    message.setdefault("model", "unknown")
    message.setdefault("stop_reason", "end_turn")
    message.setdefault("stop_sequence", None)
    usage = dict(message["usage"]) if isinstance(message.get("usage"), dict) else {}
    usage.setdefault("input_tokens", 0)
    usage.setdefault("output_tokens", 0)
    message["usage"] = usage
    return message, error, complete
