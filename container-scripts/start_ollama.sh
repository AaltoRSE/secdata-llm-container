#!/bin/bash
# Ollama API on 127.0.0.1 — open from VDI browser only if you tunnel/proxy; typically
# Continue in code-server talks to http://127.0.0.1:11434 on the same machine.
set -euo pipefail

: "${OLLAMA_HOST:=127.0.0.1:11434}"
: "${OLLAMA_MODELS:=/ollama_models}"

export OLLAMA_HOST
export OLLAMA_MODELS
mkdir -p "${OLLAMA_MODELS}"

exec ollama serve
