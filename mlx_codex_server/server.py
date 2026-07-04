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
from threading import Lock
from typing import Any
from urllib.parse import urlparse


LOG = logging.getLogger("mlx-codex-server")


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


def response_input_to_messages(body: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})

    value = body.get("input", "")
    if isinstance(value, str):
        messages.append({"role": "user", "content": value})
        return messages

    if not isinstance(value, list):
        messages.append({"role": "user", "content": str(value)})
        return messages

    for item in value:
        if not isinstance(item, dict):
            messages.append({"role": "user", "content": str(item)})
            continue

        item_type = item.get("type")
        if item_type in {"function_call", "custom_tool_call", "function_call_output", "custom_tool_call_output"}:
            messages.append({"role": "user", "content": tool_transcript_text(item)})
            continue
        if item_type in {"reasoning", "tool_search_call", "web_search_call", "image_generation_call"}:
            continue

        role = normalize_role(item.get("role"))
        content = text_from_content(item.get("content") or item.get("text") or item.get("input"))
        if content:
            messages.append({"role": role, "content": content})

    return messages


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    cached_tokens: int
    output_tokens: int


class MockEngine:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.cache_hits: dict[str, int] = {}

    def generate(self, body: dict[str, Any]) -> GenerationResult:
        key = body.get("prompt_cache_key") or "default"
        cached = self.cache_hits.get(key, 0)
        self.cache_hits[key] = 16
        return GenerationResult("OK", prompt_tokens=16, cached_tokens=cached, output_tokens=1)

    def stream(self, body: dict[str, Any]):
        result = self.generate(body)
        yield "OK", result


class MLXEngine:
    def __init__(
        self,
        model_path: str,
        model_id: str,
        *,
        prompt_cache_size: int,
        max_kv_size: int | None,
        prefill_step_size: int,
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
        self.caches: dict[str, Any] = {}
        LOG.info("model loaded")

    def cache_for(self, key: str):
        cache = self.caches.get(key)
        if cache is None:
            cache = self.cache_cls(self.prompt_cache_size)
            self.caches[key] = cache
        return cache

    def render_prompt(self, body: dict[str, Any]) -> str:
        messages = response_input_to_messages(body)
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def encode(self, prompt: str) -> list[int]:
        encoded = self.tokenizer.encode(prompt)
        if hasattr(encoded, "tolist"):
            return encoded.tolist()
        return list(encoded)

    def generate(self, body: dict[str, Any]) -> GenerationResult:
        text = "".join(piece for piece, _ in self.stream(body))
        result = getattr(self, "_last_result")
        return GenerationResult(text, result.prompt_tokens, result.cached_tokens, result.output_tokens)

    def stream(self, body: dict[str, Any]):
        patch_transformers_for_mlx_lm()
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache

        prompt = self.render_prompt(body)
        tokens = self.encode(prompt)
        key = str(body.get("prompt_cache_key") or "default")
        cache_store = self.cache_for(key)
        cache, rest = cache_store.fetch_nearest_cache(self.model_id, tokens)
        cached_tokens = len(tokens) - len(rest)
        if cache is not None and not rest:
            LOG.info("full prompt cache hit cannot be reused directly by stream_generate; falling back to prefill")
            cache = None
            cached_tokens = 0
            rest = tokens
        if cache is None:
            cache = make_prompt_cache(self.model, max_kv_size=self.max_kv_size)

        max_tokens = int(body.get("max_output_tokens") or body.get("max_tokens") or 512)
        generated: list[str] = []
        generated_token_ids: list[int] = []

        LOG.info(
            "request prompt_cache_key=%r prompt_tokens=%d cached_tokens=%d suffix_tokens=%d",
            key,
            len(tokens),
            cached_tokens,
            len(rest),
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
            if text:
                generated.append(text)
                result = GenerationResult("", len(tokens), cached_tokens, len(generated_token_ids))
                yield text, result
            finish_reason = getattr(chunk, "finish_reason", None)
            if finish_reason:
                break

        cache_key = tokens + generated_token_ids
        cache_store.insert_cache(self.model_id, copy.deepcopy(cache_key), cache)
        self._last_result = GenerationResult(
            "".join(generated),
            len(tokens),
            cached_tokens,
            len(generated_token_ids),
        )


class App:
    def __init__(self, engine: MockEngine | MLXEngine, model_id: str):
        self.engine = engine
        self.model_id = model_id
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

    def response_payload(self, rid: str, text: str, result: GenerationResult) -> dict[str, Any]:
        return {
            "id": rid,
            "object": "response",
            "created_at": now(),
            "status": "completed",
            "model": self.app.model_id,
            "output": [
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
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
        with self.app.generation_lock:
            result = self.app.engine.generate(body)
        self.write_json(self.response_payload(rid, result.text, result))

    def handle_stream_response(self, body: dict[str, Any]) -> None:
        rid = response_id()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()

        self.wfile.write(sse({"type": "response.created", "response": {"id": rid, "status": "in_progress"}}))
        self.wfile.flush()
        text_parts: list[str] = []
        last_result = GenerationResult("", 0, 0, 0)
        output_index = 0

        with self.app.generation_lock:
            for text, result in self.app.engine.stream(body):
                text_parts.append(text)
                last_result = result
                self.wfile.write(
                    sse(
                        {
                            "type": "response.output_text.delta",
                            "item_id": f"msg_{rid}",
                            "output_index": output_index,
                            "content_index": 0,
                            "delta": text,
                        }
                    )
                )
                self.wfile.flush()

        payload = self.response_payload(rid, "".join(text_parts), last_result)
        self.wfile.write(sse({"type": "response.completed", "response": payload}))
        self.wfile.write(sse("[DONE]"))
        self.wfile.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLX OpenAI Responses server for Codex")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1234)
    parser.add_argument("--model-path", default="/Users/ton/.cache/lm-studio/models/66Ton99/gpt-oss-120b")
    parser.add_argument("--model-id", default="66ton99/gpt-oss-120b")
    parser.add_argument("--prompt-cache-size", type=int, default=8)
    parser.add_argument("--max-kv-size", type=int)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--mock", action="store_true", help="Run without loading MLX model")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")

    if args.mock:
        engine: MockEngine | MLXEngine = MockEngine(args.model_id)
    else:
        engine = MLXEngine(
            args.model_path,
            args.model_id,
            prompt_cache_size=args.prompt_cache_size,
            max_kv_size=args.max_kv_size,
            prefill_step_size=args.prefill_step_size,
        )

    httpd = HTTPServer((args.host, args.port), Handler)
    httpd.app = App(engine, args.model_id)  # type: ignore[attr-defined]
    LOG.info("serving %s on http://%s:%s", args.model_id, args.host, args.port)
    httpd.serve_forever()
