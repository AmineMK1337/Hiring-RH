#!/usr/bin/env bash
# Hiring-RH Talent Orbit: one-command start on macOS / Linux.
set -e
cd "$(dirname "$0")"
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
python -m pip install -q -r requirements.txt
[ -f .env ] || { cp .env.example .env; echo "Created .env from .env.example. Edit it to choose your LLM."; }
echo "Talent Orbit: http://localhost:8000   (Ctrl+C to stop)"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
