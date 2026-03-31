#!/bin/bash
# code-server (VS Code in browser) + pre-bundled extensions (e.g. Continue).
# Listens on http://127.0.0.1:$CODE_SERVER_PORT (default 9090).
set -euo pipefail

CONTINUE_DIR="${HOME}/.continue"
mkdir -p "${CONTINUE_DIR}"
if [ ! -f "${CONTINUE_DIR}/config.yaml" ]; then
  cp /opt/dev/continue-config.yaml "${CONTINUE_DIR}/config.yaml"
fi

: "${CODE_SERVER_PORT:=9090}"
: "${CODE_SERVER_AUTH:=none}"

# For password auth: export CODE_SERVER_AUTH=password and PASSWORD='your-secret'
export PASSWORD="${PASSWORD:-}"

exec code-server \
  --bind-addr "127.0.0.1:${CODE_SERVER_PORT}" \
  --auth "${CODE_SERVER_AUTH}" \
  --extensions-dir /opt/code-server/extensions \
  "$@"
