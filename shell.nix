{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  packages = [
    pkgs.python312
    pkgs.uv
    pkgs.curl
    pkgs.jq
  ];

  shellHook = ''
    export PYTHONUNBUFFERED=1
    export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
    export UV_LINK_MODE="''${UV_LINK_MODE:-copy}"
    export MLX_CODEX_MODEL_PATH="''${MLX_CODEX_MODEL_PATH:-/Users/ton/.cache/lm-studio/models/66Ton99/gpt-oss-120b}"
    export MLX_CODEX_MODEL_ID="''${MLX_CODEX_MODEL_ID:-66ton99/gpt-oss-120b}"
    export MLX_CODEX_HOST="''${MLX_CODEX_HOST:-127.0.0.1}"
    export MLX_CODEX_PORT="''${MLX_CODEX_PORT:-1234}"

    if [ ! -x .venv/bin/python ]; then
      uv venv --python python3 .venv
    fi

    if [ ! -f .venv/.requirements.txt ] || ! cmp -s requirements.txt .venv/.requirements.txt; then
      uv pip install --python .venv/bin/python -r requirements.txt
      cp requirements.txt .venv/.requirements.txt
    fi

    export PATH="$PWD/.venv/bin:$PATH"

    echo "mlx-codex-server shell"
    echo "  run:  ./scripts/run-server"
    echo "  test: ./scripts/smoke-test --mock"
  '';
}
