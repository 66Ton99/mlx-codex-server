# mlx-codex-server

Local OpenAI Responses-compatible HTTP server for running Codex against the MLX
copy of [`66ton99/gpt-oss-120b`](https://huggingface.co/66ton99/gpt-oss-120b),
with local `prompt_cache_key` buckets and optional Codex/gpt-oss adaptation
helpers.

## Quick Start

```sh
nix-shell
./scripts/run-server --adaptation
```

The default endpoint is:

```text
http://127.0.0.1:1234/v1
```

Useful options:

```sh
./scripts/run-server --adaptation --debug
./scripts/run-server --adaptation --prompt-cache-size 4
./scripts/run-server --adaptation --max-kv-size 4096
./scripts/run-server --adaptation --stream-heartbeat-interval 15
```

Run the lightweight mock smoke test:

```sh
./scripts/smoke-test --mock
```

## Codex Provider

Example Codex provider config:

```toml
[model_providers.mlx-codex]
name = "mlx-codex"
base_url = "http://127.0.0.1:1234/v1"
wire_api = "responses"
```

Use one active Codex generation at a time for the 120B model on a single Mac.

## License

This project is licensed under the GNU General Public License v3.0 or later.
See [LICENSE](LICENSE).
