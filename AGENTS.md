# mlx-codex-server

This project runs a local OpenAI Responses-compatible HTTP server for the MLX
copy of `66ton99/gpt-oss-120b`.

The point of this server is different from LM Studio: it handles
`prompt_cache_key` locally by keeping separate in-memory MLX prompt-cache
buckets per key. Codex can send the same `prompt_cache_key` repeatedly, and this
server routes those requests to the same local prompt-cache namespace.

## Files

- `shell.nix` - the required entry point. Use `nix-shell`, not `nix-env`.
- `requirements.txt` - pinned Python wheel dependencies for MLX and MLX-LM.
- `scripts/run-server` - starts the server.
- `scripts/smoke-test` - starts a temporary server and verifies `/health`,
  `/v1/models`, and `/v1/responses`.
- `mlx_codex_server/server.py` - stdlib HTTP server plus MLX generation engine.

The local Python environment lives in `.venv/` and is intentionally ignored by
git. `nix-shell` creates and syncs it automatically.

## Setup

```sh
cd /Volumes/SRC/mlx-codex-server
nix-shell
```

The shell sets these defaults:

```sh
MLX_CODEX_MODEL_PATH=/Users/ton/.cache/lm-studio/models/66Ton99/gpt-oss-120b
MLX_CODEX_MODEL_ID=66ton99/gpt-oss-120b
MLX_CODEX_HOST=127.0.0.1
MLX_CODEX_PORT=1234
```

## Verified Launch

The lightweight launch path has been verified:

```sh
./scripts/smoke-test --mock
```

That command starts the server, calls `/v1/models`, calls `/v1/responses` with a
`prompt_cache_key`, checks that output text is returned, then stops the server.

The real MLX model path has also been verified on this machine. The server
loaded `/Users/ton/.cache/lm-studio/models/66Ton99/gpt-oss-120b`, returned
`/health`, and completed a minimal `/v1/responses` generation request on port
`18001`.

## Real Model Launch

Before launching the real 120B model, unload the same model from LM Studio. Do
not keep LM Studio and this server serving `gpt-oss-120b` at the same time unless
you deliberately want two copies of the model resident in memory.

```sh
cd /Volumes/SRC/mlx-codex-server
nix-shell
./scripts/run-server
```

Useful variants:

```sh
MLX_CODEX_PORT=18000 ./scripts/run-server
./scripts/run-server --adaptation
./scripts/run-server --prompt-cache-size 4
./scripts/run-server --max-kv-size 4096
./scripts/run-server --log-level DEBUG
```

Health check:

```sh
curl -fsS http://127.0.0.1:1234/health | jq .
curl -fsS http://127.0.0.1:1234/v1/models | jq .
```

Manual Responses API check:

```sh
curl -fsS http://127.0.0.1:1234/v1/responses \
  -H 'content-type: application/json' \
  -d '{
    "model": "66ton99/gpt-oss-120b",
    "prompt_cache_key": "codex-main",
    "input": "Reply only: OK",
    "max_output_tokens": 4,
    "stream": false
  }' | jq .
```

## Codex Config

Add a separate provider in `/Users/ton/.codex/config.toml` when you want Codex to
talk to this server:

```toml
[model_providers.mlx-codex]
name = "mlx-codex"
base_url = "http://127.0.0.1:1234/v1"
wire_api = "responses"
```

Then use a profile like:

```toml
model = "66ton99/gpt-oss-120b"
model_provider = "mlx-codex"
model_catalog_json = "/Users/ton/.codex/model-catalogs/with-lmstudio-gpt-oss-120b.json"
model_reasoning_effort = "high"
include_permissions_instructions = false
include_apps_instructions = false
include_collaboration_mode_instructions = false
include_environment_context = false

[skills]
include_instructions = false
```

Keep one active Codex generation at a time for this model. A 120B MLX model on
one Mac should be treated as single-flight: parallel generation requests usually
increase latency and memory pressure more than they help.

## Prompt Cache Behavior

`prompt_cache_key` is mapped to an in-memory `LRUPromptCache` bucket.

Cache hits still require the new prompt to share an identical token prefix with a
previous request in the same bucket. The first request with a key performs a full
prefill; later requests with the same long prefix can reuse cached KV state.

After generation, the cache is indexed by `prompt_tokens + generated_token_ids`.
That matters for Codex because later turns often include the previous assistant
answer in the next prompt.

The response includes:

```json
"usage": {
  "input_tokens_details": {
    "cached_tokens": 0
  }
}
```

`cached_tokens` becomes non-zero only after a reusable prefix exists.

The cache is process-local and memory-only. Restarting the server clears it.

Exact full-prompt hits are handled conservatively. MLX-LM `stream_generate`
requires a non-empty prompt suffix, so if the nearest cache consumes the entire
new prompt, this server falls back to full prefill instead of returning a 500.

## Compatibility Notes

MLX-LM `0.31.3` currently imports against `transformers` 5 with a legacy
tokenizer registration call. `mlx_codex_server/server.py` applies a small runtime
compatibility shim before importing MLX-LM so the server can keep the newer
MLX-LM needed for `gpt-oss` support.

The server intentionally implements the narrow API surface Codex needs:

- `GET /health`
- `GET /v1/models`
- `POST /v1/responses`
- streaming and non-streaming text responses

Unsupported request fields are ignored unless MLX generation needs them. Tool
execution is not implemented here; Codex still owns tool orchestration.

## Troubleshooting

If the port is already in use:

```sh
MLX_CODEX_PORT=18000 ./scripts/run-server
```

If Python modules are missing, re-enter the shell:

```sh
cd /Volumes/SRC/mlx-codex-server
nix-shell
```

If the real model launch runs out of memory, unload the model in LM Studio first,
reduce prompt-cache slots, or use a smaller model:

```sh
./scripts/run-server --prompt-cache-size 2
```

If Codex still logs `prompt_cache_key` as unsupported, it is not talking to this
server. Check the active Codex profile and `base_url`; LM Studio on port `1234`
does not support this local cache-key behavior.
