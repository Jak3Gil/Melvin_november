#!/bin/bash
# Start Melvin Dashboard
# Usage: ./start_dashboard.sh [brain_path] [port]

BRAIN_PATH="${1:-/tmp/melvin_brain.m}"
PORT="${2:-8080}"

echo "Starting Melvin Dashboard..."
echo "Brain: $BRAIN_PATH"
echo "Port: $PORT"
echo ""

cd "$(dirname "$0")/.."

python3 tools/melvin_dashboard.py --brain "$BRAIN_PATH" --port "$PORT" --host 0.0.0.0

