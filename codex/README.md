# Codex local profile

These files are versioned examples for running Codex through
`mlx-codex-server` instead of an LM Studio-compatible chat endpoint.

Use the `mlx-codex` provider when you want:

- OpenAI Responses API wiring
- local `prompt_cache_key` buckets
- Codex/gpt-oss adaptation helpers
- server-side visible Markdown normalization

## Files

- `mlx-codex-gpt-oss-120b.config.toml` - Codex profile template.
- `model-catalogs/with-mlx-codex-gpt-oss-120b.json` - model catalog for this profile.

## Install

Copy both files into `CODEX_HOME` or point your local launcher at the files in
this repository.

If you copy the profile into `CODEX_HOME`, keep the catalog under
`CODEX_HOME/model-catalogs/` and use a relative catalog path:

```toml
model_catalog_json = "model-catalogs/with-mlx-codex-gpt-oss-120b.json"
```

The provider must exist in your main Codex config:

```toml
[model_providers.mlx-codex]
name = "mlx-codex"
base_url = "http://127.0.0.1:1234/v1"
wire_api = "responses"
```

## Usage

Start the server first:

```sh
nix-shell
./scripts/run-server --adaptation
```

Then launch Codex from any project directory, including another workspace such as
a Bitburner scripts folder, using a profile whose `model_provider` is
`mlx-codex`. Keep machine-specific paths out of committed profiles; use relative
paths or local symlinks from `CODEX_HOME` when a launcher needs a stable filename.

Do not use an LM Studio profile for this server. It can still reach port `1234`,
but it does not use this project's Responses API behavior.
