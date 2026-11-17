#!/usr/bin/env sh
# One-command local run: builds the React app and serves everything from FastAPI on :8000
set -e
cd "$(dirname "$0")/frontend" && npm install && npm run build
cd ../backend && . .venv/bin/activate && uvicorn api.main:app --port 8000
