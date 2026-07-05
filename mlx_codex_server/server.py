from __future__ import annotations

import argparse
import copy
import json
import logging
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Event
from threading import Lock
from threading import Thread
from typing import Any
from urllib.parse import urlparse


LOG = logging.getLogger("mlx-codex-server")
DEBUG_PREVIEW_CHARS = 4000


def patch_transformers_for_mlx_lm() -> None:
    """Tolerate mlx-lm's legacy string tokenizer registration on transformers 5."""
    try:
        from transformers import AutoTokenizer
    except Exception:
        return

    if getattr(AutoTokenizer.register, "_mlx_codex_patched", False):
        return

    original_register = AutoTokenizer.register

    def register(config_class, *args, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(config_class, str):
            LOG.debug("ignoring legacy AutoTokenizer.register(%r)", config_class)
            return None
        return original_register(config_class, *args, **kwargs)

    register._mlx_codex_patched = True  # type: ignore[attr-defined]
    AutoTokenizer.register = register  # type: ignore[method-assign]


def now() -> int:
    return int(time.time())


def response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def sse(data: dict[str, Any] | str) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n".encode("utf-8")


def preview(value: Any, *, limit: int = DEBUG_PREVIEW_CHARS) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... <truncated {len(text) - limit} chars>"


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"input_text", "output_text", "text"}:
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return text_from_content([content])
    return "" if content is None else str(content)


def normalize_role(role: str | None) -> str:
    if role in {"developer", "system"}:
        return "system"
    if role in {"assistant", "user"}:
        return role
    return "user"


def tool_transcript_text(item: dict[str, Any]) -> str:
    item_type = str(item.get("type") or "tool")
    call_id = item.get("call_id")
    name = item.get("name")

    if item_type in {"function_call_output", "custom_tool_call_output"}:
        content = text_from_content(item.get("output") or item.get("content"))
        label = "tool result"
    else:
        content = text_from_content(item.get("arguments") or item.get("input") or item.get("content"))
        label = "tool call"

    details = ", ".join(str(part) for part in (name, call_id) if part)
    prefix = f"{label} ({details})" if details else label
    return f"[{prefix}]\n{content}" if content else f"[{prefix}]"


def append_message(messages: list[dict[str, str]], role: str, content: str, *, adaptation: bool) -> None:
    if adaptation:
        content = sanitize_message_content(role, content)
    if content:
        messages.append({"role": role, "content": content})


def sanitize_message_content(role: str, content: str) -> str:
    if role == "assistant":
        content = response_text_from_generation(content)
    return escape_harmony_tokens(content)


def escape_harmony_tokens(content: str) -> str:
    for token in ("<|channel|>", "<|message|>", "<|start|>", "<|end|>"):
        content = content.replace(token, token.replace("<|", "< |"))
    return content


def response_input_to_messages(body: dict[str, Any], *, adaptation: bool = False) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        append_message(messages, "system", instructions, adaptation=adaptation)

    value = body.get("input", "")
    if isinstance(value, str):
        append_message(messages, "user", value, adaptation=adaptation)
        return messages

    if not isinstance(value, list):
        append_message(messages, "user", str(value), adaptation=adaptation)
        return messages

    for item in value:
        if not isinstance(item, dict):
            append_message(messages, "user", str(item), adaptation=adaptation)
            continue

        item_type = item.get("type")
        if item_type in {"function_call", "custom_tool_call", "function_call_output", "custom_tool_call_output"}:
            append_message(messages, "user", tool_transcript_text(item), adaptation=adaptation)
            continue
        if item_type in {"reasoning", "tool_search_call", "web_search_call", "image_generation_call"}:
            continue

        role = normalize_role(item.get("role"))
        content = text_from_content(item.get("content") or item.get("text") or item.get("input"))
        if content:
            append_message(messages, role, content, adaptation=adaptation)

    LOG.debug("normalized %d response input item(s) into %d chat message(s)", len(value), len(messages))
    return messages


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    cached_tokens: int
    output_tokens: int
    max_tokens: int = 0


def parse_tool_call_text(text: str) -> tuple[str, str] | None:
    marker = "to="
    message_marker = "<|message|>"
    end_marker = "<|end|>"
    search_from = 0
    found: tuple[str, str] | None = None

    while True:
        marker_index = text.find(marker, search_from)
        if marker_index == -1:
            return found or parse_bracketed_tool_call_text(text)
        message_index = text.find(message_marker, marker_index)
        if message_index == -1:
            return found or parse_bracketed_tool_call_text(text)

        name_start = marker_index + len(marker)
        name_end = name_start
        while name_end < message_index and (text[name_end].isalnum() or text[name_end] in "._-"):
            name_end += 1
        name = text[name_start:name_end]
        if not name:
            search_from = marker_index + len(marker)
            continue

        content_start = message_index + len(message_marker)
        content_end = text.find(end_marker, content_start)
        if content_end == -1:
            content_end = len(text)
        found = (name, text[content_start:content_end])
        search_from = content_end + len(end_marker)


def parse_bracketed_tool_call_text(text: str) -> tuple[str, str] | None:
    marker = "[tool call ("
    marker_index = text.rfind(marker)
    if marker_index == -1:
        return None
    name_start = marker_index + len(marker)
    header_end = text.find(")]", name_start)
    if header_end == -1:
        return None
    header = text[name_start:header_end]
    name = header.split(",", 1)[0].strip()
    if not name:
        return None

    arguments_start = text.find("{", header_end)
    arguments_end = text.rfind("}")
    if arguments_start == -1 or arguments_end == -1 or arguments_end < arguments_start:
        return None
    return name, text[arguments_start : arguments_end + 1]


def parse_harmony_messages(text: str) -> list[tuple[str | None, str]]:
    start_marker = "<|start|>assistant"
    channel_marker = "<|channel|>"
    message_marker = "<|message|>"
    end_marker = "<|end|>"
    messages: list[tuple[str | None, str]] = []
    search_from = 0

    while True:
        start_index = text.find(start_marker, search_from)
        if start_index == -1:
            return messages
        channel_index = text.find(channel_marker, start_index)
        message_index = text.find(message_marker, start_index)
        if message_index == -1:
            return messages

        channel = None
        if channel_index != -1 and channel_index < message_index:
            channel_start = channel_index + len(channel_marker)
            channel_end = channel_start
            while channel_end < message_index and (text[channel_end].isalnum() or text[channel_end] in "._-"):
                channel_end += 1
            channel = text[channel_start:channel_end] or None

        content_start = message_index + len(message_marker)
        content_end = text.find(end_marker, content_start)
        if content_end == -1:
            content_end = len(text)
        messages.append((channel, text[content_start:content_end]))
        search_from = content_end + len(end_marker)


def response_text_from_generation(text: str) -> str:
    messages = parse_harmony_messages(text)
    if not messages:
        return text
    for preferred_channel in ("final", "commentary"):
        for channel, content in reversed(messages):
            if channel == preferred_channel:
                return content.strip()
    for channel, content in reversed(messages):
        if channel != "analysis":
            return content.strip()
    return "[local model produced analysis-only output without a final answer or tool call]"


def tools_for_chat_template(body: dict[str, Any]) -> list[dict[str, Any]]:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return []
    normalized = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            normalized.append({"type": "function", "function": function})
            continue

        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        description = str(tool.get("description") or f"Call {name}.")
        parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Tool input.",
                    }
                },
                "required": ["input"],
            }
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return normalized


def tool_info_for(body: dict[str, Any], name: str) -> tuple[str, str | None]:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return inferred_tool_info(name)
    short_name = name.rsplit(".", 1)[-1]
    for tool in tools:
        if isinstance(tool, dict) and tool.get("name") in {name, short_name}:
            tool_type = tool.get("type")
            tool_name = str(tool.get("name"))
            return tool_name, str(tool_type) if tool_type is not None else None
    return inferred_tool_info(name)


def inferred_tool_info(name: str) -> tuple[str, str | None]:
    short_name = name.rsplit(".", 1)[-1]
    if short_name in {"exec_command", "write_stdin", "update_plan"}:
        return short_name, "function"
    return short_name, None


def tool_arguments_for(tool_name: str, tool_type: str | None, raw: str) -> str:
    if tool_type == "function":
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = repair_jsonish_tool_arguments(raw)
        return json.dumps(parsed, ensure_ascii=False)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(parsed, dict) and isinstance(parsed.get("input"), str):
        return parsed["input"]
    if tool_name in {"exec_command", "write_stdin", "update_plan"}:
        return json.dumps(parsed, ensure_ascii=False)
    if isinstance(parsed, dict) and isinstance(parsed.get("patch"), str):
        return parsed["patch"]
    return raw


def repair_jsonish_tool_arguments(raw: str) -> dict[str, str]:
    text = raw.strip()
    for key in ("cmd", "command"):
        prefix = f'{{"{key}":"'
        suffix = '"}'
        if text.startswith(prefix) and text.endswith(suffix):
            return {key: text[len(prefix) : -len(suffix)]}
    return {"input": raw}


class MockEngine:
    def __init__(self, model_id: str, *, adaptation: bool):
        self.model_id = model_id
        self.adaptation = adaptation
        self.cache_hits: dict[str, int] = {}

    def generate(self, body: dict[str, Any]) -> GenerationResult:
        key = body.get("prompt_cache_key") or "default"
        cached = self.cache_hits.get(key, 0)
        self.cache_hits[key] = 16
        if self.adaptation and "MOCK_APPLY_PATCH_CALL" in json.dumps(body.get("input", ""), ensure_ascii=False):
            text = (
                '<|start|>assistant<|channel|>commentary to=apply_patch code<|message|>'
                '{"patch":"*** Begin Patch\\n*** Add File: mock.txt\\n+OK\\n*** End Patch\\n"}'
                '<|end|>'
            )
            return GenerationResult(text, prompt_tokens=16, cached_tokens=cached, output_tokens=16, max_tokens=32)
        if self.adaptation and "MOCK_FINAL_HARMONY" in json.dumps(body.get("input", ""), ensure_ascii=False):
            text = (
                "<|channel|>analysis<|message|>Hidden reasoning.<|end|>"
                "<|start|>assistant<|channel|>final<|message|>Clean final answer.<|end|>"
            )
            return GenerationResult(text, prompt_tokens=16, cached_tokens=cached, output_tokens=16, max_tokens=32)
        if self.adaptation and "MOCK_EXEC_COMMAND_BRACKET" in json.dumps(body.get("input", ""), ensure_ascii=False):
            text = (
                "<|channel|>analysis<|message|>Need to commit.<|end|>"
                "<|start|>assistant<|channel|>commentary<|message|>"
                "[tool call (exec_command, commit_money_js)]\n"
                '{"cmd":"git add money.js && git commit -m \\"Add money.js orchestrator for core money scripts\\""}'
                "<|end|>"
            )
            return GenerationResult(text, prompt_tokens=16, cached_tokens=cached, output_tokens=32, max_tokens=64)
        return GenerationResult("OK", prompt_tokens=16, cached_tokens=cached, output_tokens=1, max_tokens=4)

    def stream(self, body: dict[str, Any]):
        if "MOCK_SLOW_STREAM" in json.dumps(body.get("input", ""), ensure_ascii=False):
            time.sleep(0.15)
        result = self.generate(body)
        yield result.text, result


class MLXEngine:
    def __init__(
        self,
        model_path: str,
        model_id: str,
        *,
        prompt_cache_size: int,
        max_kv_size: int | None,
        prefill_step_size: int,
        min_output_tokens: int,
        adaptation: bool,
    ):
        patch_transformers_for_mlx_lm()
        from mlx_lm import load
        from mlx_lm.models.cache import LRUPromptCache

        LOG.info("loading MLX model %s from %s", model_id, model_path)
        self.model_id = model_id
        self.model, self.tokenizer = load(model_path)
        self.cache_cls = LRUPromptCache
        self.prompt_cache_size = prompt_cache_size
        self.max_kv_size = max_kv_size
        self.prefill_step_size = prefill_step_size
        self.min_output_tokens = min_output_tokens
        self.adaptation = adaptation
        self.caches: dict[str, Any] = {}
        LOG.info("model loaded")

    def cache_for(self, key: str):
        cache = self.caches.get(key)
        if cache is None:
            cache = self.cache_cls(self.prompt_cache_size)
            self.caches[key] = cache
        return cache

    def render_prompt(self, body: dict[str, Any]) -> str:
        messages = response_input_to_messages(body, adaptation=self.adaptation)
        tools = tools_for_chat_template(body) if self.adaptation else []
        LOG.debug("chat messages: %s", preview(messages))
        if hasattr(self.tokenizer, "apply_chat_template"):
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if tools:
                template_kwargs["tools"] = tools
            try:
                prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
            except Exception:
                if not tools:
                    raise
                LOG.exception("apply_chat_template failed with tools; retrying without tools")
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        LOG.debug("rendered prompt chars=%d preview=%s", len(prompt), preview(prompt))
        return prompt

    def encode(self, prompt: str) -> list[int]:
        encoded = self.tokenizer.encode(prompt)
        if hasattr(encoded, "tolist"):
            return encoded.tolist()
        return list(encoded)

    def generate(self, body: dict[str, Any]) -> GenerationResult:
        text = "".join(piece for piece, _ in self.stream(body))
        result = getattr(self, "_last_result")
        return GenerationResult(text, result.prompt_tokens, result.cached_tokens, result.output_tokens, result.max_tokens)

    def stream(self, body: dict[str, Any]):
        patch_transformers_for_mlx_lm()
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import can_trim_prompt_cache
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.models.cache import trim_prompt_cache

        prompt = self.render_prompt(body)
        tokens = self.encode(prompt)
        key = str(body.get("prompt_cache_key") or "default")
        cache_store = self.cache_for(key)
        cache, rest = cache_store.fetch_nearest_cache(self.model, tokens)
        cached_tokens = len(tokens) - len(rest)
        LOG.debug(
            "cache lookup key=%r hit=%s prompt_tokens=%d cached_tokens=%d suffix_tokens=%d",
            key,
            cache is not None,
            len(tokens),
            cached_tokens,
            len(rest),
        )
        if cache is not None and not rest:
            LOG.info("full prompt cache hit cannot be reused directly by stream_generate; falling back to prefill")
            cache = None
            cached_tokens = 0
            rest = tokens
        if cache is None:
            cache = make_prompt_cache(self.model, max_kv_size=self.max_kv_size)

        requested_max_tokens = int(body.get("max_output_tokens") or body.get("max_tokens") or 512)
        max_tokens = max(requested_max_tokens, self.min_output_tokens) if self.adaptation else requested_max_tokens
        generated: list[str] = []
        generated_token_ids: list[int] = []

        LOG.info(
            "request prompt_cache_key=%r prompt_tokens=%d cached_tokens=%d suffix_tokens=%d max_tokens=%d requested_max_tokens=%d",
            key,
            len(tokens),
            cached_tokens,
            len(rest),
            max_tokens,
            requested_max_tokens,
        )

        for chunk in stream_generate(
            self.model,
            self.tokenizer,
            prompt=rest,
            max_tokens=max_tokens,
            prompt_cache=cache,
            prefill_step_size=self.prefill_step_size,
        ):
            text = getattr(chunk, "text", "")
            token = getattr(chunk, "token", None)
            if isinstance(token, int):
                generated_token_ids.append(token)
                LOG.debug("generated token id=%d text=%r", token, text)
            if text:
                generated.append(text)
                result = GenerationResult("", len(tokens), cached_tokens, len(generated_token_ids), max_tokens)
                yield text, result
            finish_reason = getattr(chunk, "finish_reason", None)
            if finish_reason:
                LOG.debug("generation finish_reason=%r", finish_reason)
                break

        cache_is_trimmable = can_trim_prompt_cache(cache)
        if generated_token_ids and cache_is_trimmable:
            prompt_only_cache = copy.deepcopy(cache)
            trim_prompt_cache(prompt_only_cache, len(generated_token_ids))
            cache_store.insert_cache(self.model, copy.deepcopy(tokens), prompt_only_cache, cache_type="user")
            LOG.debug("stored prompt-only cache key=%r prompt_tokens=%d", key, len(tokens))
        elif generated_token_ids:
            LOG.debug("prompt cache key=%r is not trimmable; storing generated sequence only", key)

        cache_key = tokens + generated_token_ids
        cache_store.insert_cache(self.model, copy.deepcopy(cache_key), cache)
        LOG.debug("stored prompt cache key=%r total_cached_sequence_tokens=%d", key, len(cache_key))
        self._last_result = GenerationResult(
            "".join(generated),
            len(tokens),
            cached_tokens,
            len(generated_token_ids),
            max_tokens,
        )


class App:
    def __init__(self, engine: MockEngine | MLXEngine, model_id: str, *, adaptation: bool, stream_heartbeat_interval: float):
        self.engine = engine
        self.model_id = model_id
        self.adaptation = adaptation
        self.stream_heartbeat_interval = stream_heartbeat_interval
        self.generation_lock = Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "mlx-codex-server/0.1"

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or "0")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def write_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self.write_json({"ok": True, "model": self.app.model_id})
            return
        if path == "/v1/models":
            self.write_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.app.model_id,
                            "object": "model",
                            "created": 0,
                            "owned_by": "local",
                        }
                    ],
                }
            )
            return
        self.write_json({"error": {"message": "not found"}}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/v1/responses":
            self.write_json({"error": {"message": "not found"}}, HTTPStatus.NOT_FOUND)
            return

        try:
            body = self.read_json()
            LOG.debug(
                "request path=%s stream=%s model=%r prompt_cache_key=%r max_output_tokens=%r body=%s",
                path,
                body.get("stream"),
                body.get("model"),
                body.get("prompt_cache_key"),
                body.get("max_output_tokens") or body.get("max_tokens"),
                preview(body),
            )
            if body.get("stream"):
                self.handle_stream_response(body)
            else:
                self.handle_response(body)
        except Exception as exc:
            LOG.exception("request failed")
            self.write_json(
                {"error": {"message": str(exc), "type": "server_error"}},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def response_output_item(self, rid: str, body: dict[str, Any], text: str) -> dict[str, Any]:
        tool_call = parse_tool_call_text(text) if self.app.adaptation else None
        if tool_call:
            name, raw_arguments = tool_call
            call_id = f"call_{uuid.uuid4().hex}"
            tool_name, tool_type = tool_info_for(body, name)
            arguments = tool_arguments_for(tool_name, tool_type, raw_arguments)
            if tool_type == "function":
                return {
                    "id": f"fc_{uuid.uuid4().hex}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": tool_name,
                    "arguments": arguments,
                }
            return {
                "id": f"ctc_{uuid.uuid4().hex}",
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": call_id,
                "name": tool_name,
                "input": arguments,
            }
        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": response_text_from_generation(text) if self.app.adaptation else text,
                    "annotations": [],
                }
            ],
        }

    def response_payload(self, rid: str, body: dict[str, Any], text: str, result: GenerationResult) -> dict[str, Any]:
        return {
            "id": rid,
            "object": "response",
            "created_at": now(),
            "status": "completed",
            "model": self.app.model_id,
            "output": [self.response_output_item(rid, body, text)],
            "usage": {
                "input_tokens": result.prompt_tokens,
                "input_tokens_details": {"cached_tokens": result.cached_tokens},
                "output_tokens": result.output_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": result.prompt_tokens + result.output_tokens,
            },
        }

    def handle_response(self, body: dict[str, Any]) -> None:
        rid = response_id()
        LOG.debug("starting non-stream response id=%s", rid)
        with self.app.generation_lock:
            result = self.app.engine.generate(body)
        LOG.debug(
            "completed non-stream response id=%s input_tokens=%d cached_tokens=%d output_tokens=%d text=%s",
            rid,
            result.prompt_tokens,
            result.cached_tokens,
            result.output_tokens,
            preview(result.text),
        )
        self.write_json(self.response_payload(rid, body, result.text, result))

    def handle_stream_response(self, body: dict[str, Any]) -> None:
        rid = response_id()
        LOG.debug("starting stream response id=%s", rid)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()

        write_lock = Lock()

        def write_sse(data: bytes) -> None:
            with write_lock:
                self.wfile.write(data)
                self.wfile.flush()

        write_sse(sse({"type": "response.created", "response": {"id": rid, "status": "in_progress"}}))
        text_parts: list[str] = []
        last_result = GenerationResult("", 0, 0, 0)
        requested_max_tokens = int(body.get("max_output_tokens") or body.get("max_tokens") or 512)
        engine_min_output_tokens = int(getattr(self.app.engine, "min_output_tokens", requested_max_tokens))
        expected_max_tokens = max(requested_max_tokens, engine_min_output_tokens) if self.app.adaptation else requested_max_tokens
        progress = {
            "phase": "prefill",
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "max_tokens": expected_max_tokens,
        }
        progress_lock = Lock()
        started_at = time.monotonic()

        stop_heartbeat = Event()

        def heartbeat() -> None:
            while not stop_heartbeat.wait(self.app.stream_heartbeat_interval):
                with progress_lock:
                    phase = str(progress["phase"])
                    prompt_tokens = int(progress["prompt_tokens"])
                    cached_tokens = int(progress["cached_tokens"])
                    output_tokens = int(progress["output_tokens"])
                    max_tokens = int(progress["max_tokens"])
                percent = 0 if phase == "prefill" else min(99, int(output_tokens * 100 / max(max_tokens, 1)))
                elapsed = time.monotonic() - started_at
                LOG.debug(
                    "stream response id=%s heartbeat progress=%d%% phase=%s elapsed=%.1fs prompt_tokens=%d cached_tokens=%d output_tokens=%d/%d",
                    rid,
                    percent,
                    phase,
                    elapsed,
                    prompt_tokens,
                    cached_tokens,
                    output_tokens,
                    max_tokens,
                )
                try:
                    write_sse(
                        (
                            f": keep-alive progress={percent}% phase={phase} elapsed={elapsed:.1f}s "
                            f"output_tokens={output_tokens}/{max_tokens}\n\n"
                        ).encode()
                    )
                except OSError:
                    stop_heartbeat.set()
                    return

        heartbeat_worker = Thread(target=heartbeat, name=f"mlx-heartbeat-{rid}", daemon=True)
        heartbeat_worker.start()

        try:
            with self.app.generation_lock:
                for text, result in self.app.engine.stream(body):
                    text_parts.append(text)
                    last_result = result
                    with progress_lock:
                        progress["phase"] = "generating"
                        progress["prompt_tokens"] = result.prompt_tokens
                        progress["cached_tokens"] = result.cached_tokens
                        progress["output_tokens"] = result.output_tokens
                        progress["max_tokens"] = result.max_tokens
                    LOG.debug("stream response id=%s delta=%r output_tokens=%d", rid, text, result.output_tokens)
        finally:
            stop_heartbeat.set()
            heartbeat_worker.join(timeout=1)

        payload = self.response_payload(rid, body, "".join(text_parts), last_result)
        output_item = payload["output"][0]
        LOG.debug(
            "completed stream response id=%s input_tokens=%d cached_tokens=%d output_tokens=%d text=%s",
            rid,
            last_result.prompt_tokens,
            last_result.cached_tokens,
            last_result.output_tokens,
            preview("".join(text_parts)),
        )
        write_sse(sse({"type": "response.output_item.added", "output_index": 0, "item": output_item}))
        write_sse(sse({"type": "response.output_item.done", "output_index": 0, "item": output_item}))
        write_sse(sse({"type": "response.completed", "response": payload}))
        write_sse(sse("[DONE]"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLX OpenAI Responses server for Codex")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--model-path", default="/Users/ton/.cache/lm-studio/models/66Ton99/gpt-oss-120b")
    parser.add_argument("--model-id", default="66ton99/gpt-oss-120b")
    parser.add_argument("--prompt-cache-size", type=int, default=8)
    parser.add_argument("--max-kv-size", type=int)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--min-output-tokens", type=int, default=2048)
    parser.add_argument("--stream-heartbeat-interval", type=float, default=15.0)
    parser.add_argument("-a", "--adaptation", action="store_true", help="Enable Codex/gpt-oss compatibility adaptations")
    parser.add_argument("--mock", action="store_true", help="Run without loading MLX model")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable verbose debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_level = "DEBUG" if args.debug else args.log_level.upper()
    logging.basicConfig(level=getattr(logging, log_level), format="%(asctime)s %(levelname)s %(message)s")
    if args.debug:
        LOG.debug("debug logging enabled")
    LOG.info("adaptation mode %s", "enabled" if args.adaptation else "disabled")

    if args.mock:
        engine: MockEngine | MLXEngine = MockEngine(args.model_id, adaptation=args.adaptation)
    else:
        engine = MLXEngine(
            args.model_path,
            args.model_id,
            prompt_cache_size=args.prompt_cache_size,
            max_kv_size=args.max_kv_size,
            prefill_step_size=args.prefill_step_size,
            min_output_tokens=args.min_output_tokens,
            adaptation=args.adaptation,
        )

    httpd = HTTPServer((args.host, args.port), Handler)
    httpd.app = App(
        engine,
        args.model_id,
        adaptation=args.adaptation,
        stream_heartbeat_interval=args.stream_heartbeat_interval,
    )  # type: ignore[attr-defined]
    LOG.info("serving %s on http://%s:%s", args.model_id, args.host, args.port)
    httpd.serve_forever()
