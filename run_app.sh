#!/usr/bin/env bash
# Launch the CASMI 2026 Streamlit Web App
set -e
cd "$(dirname "$0")"

echo "🧪 Starting CASMI 2026 Molecule Identifier Web Application..."
./.venv/bin/streamlit run frontend/app.py --server.port 8501
