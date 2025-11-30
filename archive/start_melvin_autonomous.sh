#!/bin/bash
# Start Melvin autonomously on Jetson
# Run this script to start Melvin in background

set -e

cd ~/melvin_system

# Check if already running
if pgrep -f "./melvin" > /dev/null; then
    echo "Melvin is already running (PID: $(pgrep -f './melvin'))"
    echo "To restart, first stop: pkill -f melvin"
    exit 0
fi

echo "Starting Melvin autonomously..."

# Start in background with nohup
nohup ./melvin > melvin.log 2>&1 &
MELVIN_PID=$!

echo "Melvin started in background"
echo "  PID: $MELVIN_PID"
echo "  Log: ~/melvin_system/melvin.log"
echo ""
echo "To check status: ps aux | grep melvin"
echo "To view logs: tail -f ~/melvin_system/melvin.log"
echo "To stop: pkill -f melvin"
echo ""

# Wait a moment and verify it's running
sleep 2
if pgrep -f "./melvin" > /dev/null; then
    echo "✓ Melvin is running successfully!"
else
    echo "✗ Melvin failed to start. Check melvin.log for errors."
    exit 1
fi


