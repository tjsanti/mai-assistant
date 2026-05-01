#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PID=""

cleanup() {
  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "Starting backend on http://localhost:8000"
(
  cd "$ROOT_DIR/backend"
  exec uv run uvicorn app.main:app --reload
) &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:5173"
cd "$ROOT_DIR/frontend"
npm run dev
