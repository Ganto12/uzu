#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
    echo "Usage: bench-mlxserve model.json" >&2
    exit 2
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BENCHMARK_MLXSERVE_BINARY:-}" ]]; then
    PYTHON=("$BENCHMARK_MODEL_PYTHON")
else
    "$PROJECT_DIR/bootstrap.sh"
    BENCHMARK_MLXSERVE_BINARY="$PROJECT_DIR/zig-out/bin/engine-mlxserve"
    PYTHON=(uv run --project "$PROJECT_DIR/.." python)
fi

exec "${PYTHON[@]}" -c '
import json, os, sys
from common import get_model_path
config = json.load(open(sys.argv[2]))
args = ["--config", sys.argv[2]]
for key, flag in [("model", "--model"), ("draft_model", "--draft-model")]:
    model = config.get(key)
    if model is not None:
        args += [flag, model if os.path.isdir(model) else get_model_path(model)]
os.execv(sys.argv[1], [sys.argv[1], *args])
' "$BENCHMARK_MLXSERVE_BINARY" "$1"
