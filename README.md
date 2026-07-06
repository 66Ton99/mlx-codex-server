# mlx-codex-server

Lightweight MLX server tuned for Codex 0.142+.

It implements the narrow OpenAI Responses API surface Codex needs to run against
the MLX copy of
[`66ton99/gpt-oss-120b`](https://huggingface.co/66ton99/gpt-oss-120b), with
local `prompt_cache_key` buckets and optional Codex/gpt-oss adaptation helpers.
It is intentionally not a general-purpose OpenAI-compatible gateway.

## Quick Start

```sh
nix-shell
export MLX_CODEX_MODEL_PATH=/path/to/local/mlx/model
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

Explicit options are saved to `config.json` and reused on later starts. Use
`--no-save-config` for one-off overrides. The `--mock` flag is never persisted.

Run the lightweight mock smoke test:

```sh
./scripts/smoke-test --mock
```

## Codex Provider

Add the Responses-compatible provider to your Codex 0.142+ config:

```toml
[model_providers.mlx-codex]
name = "mlx-codex"
base_url = "http://127.0.0.1:1234/v1"
wire_api = "responses"
```

This provider must be used instead of an LM Studio/chat-completions provider if
you want the server-side Codex adaptation helpers, prompt-cache buckets, and
visible Markdown normalization.

This repository includes a tracked Codex profile and model catalog:

```text
codex/mlx-codex-gpt-oss-120b.config.toml
codex/model-catalogs/with-mlx-codex-gpt-oss-120b.json
```

See [codex/README.md](codex/README.md) for install and usage notes.

Copy the profile into `CODEX_HOME`, then replace the placeholder catalog path
with the absolute path to this repository:

```sh
mkdir -p "$HOME/.codex/model-catalogs"
cp codex/mlx-codex-gpt-oss-120b.config.toml "$HOME/.codex/"
cp codex/model-catalogs/with-mlx-codex-gpt-oss-120b.json "$HOME/.codex/model-catalogs/"
```

Then set `model_catalog_json` in the copied profile to the copied catalog path,
for example:

```toml
model_catalog_json = "model-catalogs/with-mlx-codex-gpt-oss-120b.json"
```

Do not use a profile whose `model_provider` is `lmstudio` for this server. That
path bypasses `/v1/responses` behavior that this project implements.

Use one active Codex generation at a time for the 120B model on a single Mac.

## License

This project is licensed under the GNU General Public License v3.0 or later.
See [LICENSE](LICENSE).
