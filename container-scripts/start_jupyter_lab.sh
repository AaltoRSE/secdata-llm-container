#!/bin/bash
# JupyterLab on http://127.0.0.1:$JUPYTER_PORT (default 8888).
set -euo pipefail

: "${JUPYTER_PORT:=8888}"
: "${JUPYTER_IP:=127.0.0.1}"

exec jupyter lab --no-browser --ip="${JUPYTER_IP}" --port="${JUPYTER_PORT}" "$@"
