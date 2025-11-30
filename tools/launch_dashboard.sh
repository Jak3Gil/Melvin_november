#!/bin/bash
# Launch Melvin Dashboard as a standalone app
# Opens in a separate browser window

BRAIN_PATH="${1:-/tmp/melvin_brain.m}"
PORT="${2:-8080}"
HOST="${3:-127.0.0.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "Melvin Dashboard App"
echo "=========================================="
echo "Brain: $BRAIN_PATH"
echo "URL: http://$HOST:$PORT"
echo "=========================================="
echo ""

# Check if Electron is available
if command -v electron &> /dev/null || [ -f "node_modules/.bin/electron" ]; then
    echo "Launching as Electron app..."
    export MELVIN_BRAIN="$BRAIN_PATH"
    if [ -f "node_modules/.bin/electron" ]; then
        ./node_modules/.bin/electron tools/electron-main.js
    else
        electron tools/electron-main.js
    fi
else
    echo "Electron not found, launching as Python app with browser..."
    python3 tools/melvin_dashboard_app.py --brain "$BRAIN_PATH" --port "$PORT" --host "$HOST"
fi

