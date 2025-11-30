#!/bin/bash
#
# run_melvin.sh - Run Melvin on Jetson
#

BRAIN_FILE="${1:-~/melvin_data/brains/melvin.m}"
AUDIO_DEVICE="${2:-default}"

echo "=========================================="
echo "Starting Melvin"
echo "=========================================="
echo ""
echo "Brain file: $BRAIN_FILE"
echo "Audio device: $AUDIO_DEVICE"
echo ""

# Create brain if it doesn't exist
if [ ! -f "$BRAIN_FILE" ]; then
    echo "Creating new brain file..."
    mkdir -p "$(dirname "$BRAIN_FILE")"
    ./melvin_hardware_runner "$BRAIN_FILE" "$AUDIO_DEVICE" "$AUDIO_DEVICE" &
    sleep 2
    pkill -f melvin_hardware_runner
    echo "Brain file created"
    echo ""
fi

# Run Melvin
echo "Starting Melvin..."
./melvin_hardware_runner "$BRAIN_FILE" "$AUDIO_DEVICE" "$AUDIO_DEVICE"
