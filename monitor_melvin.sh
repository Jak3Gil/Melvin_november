#!/bin/bash
# monitor_melvin.sh - Monitor Melvin brain from Mac
# Connects to Jetson and displays live stats

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
BRAIN_PATH="/tmp/brain.m"
INTERVAL=5

# Check if brain file exists
sshpass -p "$JETSON_PASS" ssh "$JETSON_USER@$JETSON_IP" "test -f $BRAIN_PATH" || {
    echo "Error: Brain file not found at $BRAIN_PATH"
    echo "Create it first with: ./create_brain.sh"
    exit 1
}

echo "Connecting to Jetson and monitoring Melvin brain..."
echo "Press Ctrl+C to stop"
echo ""

# Run monitor on Jetson and stream output
sshpass -p "$JETSON_PASS" ssh "$JETSON_USER@$JETSON_IP" \
    "cd /mnt/melvin_ssd/melvin && ./melvin_monitor $BRAIN_PATH $INTERVAL"

