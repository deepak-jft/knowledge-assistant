#!/bin/bash
# ─────────────────────────────────────────────────────
#  run.sh — Quick-start the Knowledge Assistant
#  Usage:  bash run.sh
# ─────────────────────────────────────────────────────
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌  Virtual environment not found."
    echo "    Run:  python3 -m venv venv && venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Check .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env not found — copying from .env.example"
    cp .env.example .env
fi

echo "🚀  Starting Knowledge Assistant …"
streamlit run app.py
