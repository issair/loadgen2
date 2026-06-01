#!/bin/bash
# ------------------------------------------------------------------
#  Unified entrypoint for GLM-5.1 & DeepSeek-R1 MLPerf docker image.
#
#  Usage:
#    docker run ... glm-5.1   [args...]
#    docker run ... deepseek  [args...]
# ------------------------------------------------------------------
set -euo pipefail

PROJECT="${1:-}"
shift || true

case "$PROJECT" in
    glm-5.1|glm)
        PROJ_DIR="/workspace/language/glm-5.1"
        ;;
    deepseek|deepseek-r1|r1)
        PROJ_DIR="/workspace/language/deepseek-r1"
        ;;
    *)
        # Default to glm-5.1 for backward compatibility:
        # treat the first arg as a Python script/arg, not a project name.
        PROJ_DIR="/workspace/language/glm-5.1"
        set -- "$PROJECT" "$@"   # put the first arg back
        ;;
esac

cd "$PROJ_DIR"
exec uv run --offline python "$@"
