#!/bin/bash
# Start Melvin with visualization server enabled
# Opens web interface for graph visualization

set -e

BRAIN_FILE="${1:-melvin.m}"

if [ ! -f "$BRAIN_FILE" ]; then
    echo "Error: Brain file '$BRAIN_FILE' not found"
    echo "Run: ./init_melvin_jetson.sh first"
    exit 1
fi

# Get Jetson IP address
JETSON_IP=$(hostname -I | awk '{print $1}')

echo "=========================================="
echo "Starting Melvin with Visualization"
echo "=========================================="
echo ""
echo "Visualization will be available at:"
echo "  http://localhost:8080"
echo "  http://$JETSON_IP:8080"
echo ""
echo "The visualization updates every 500ms"
echo "Press Ctrl+C to stop"
echo ""

# Set environment variable to enable visual server
export MELVIN_ENABLE_VISUAL=1

# Run melvin
./melvin "$BRAIN_FILE"

