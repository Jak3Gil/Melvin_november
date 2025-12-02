#!/bin/bash
# jetson_terminal.sh - Open interactive terminal to Jetson via USB
# Keeps you connected to see what's happening in real-time

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"

echo "=========================================="
echo "Opening terminal connection to Jetson..."
echo "=========================================="
echo ""
echo "You'll be connected to: $JETSON_USER@$JETSON_IP"
echo "Working directory: /home/melvin/melvin"
echo ""
echo "Useful commands once connected:"
echo "  ps aux | grep melvin        # Check running processes"
echo "  tail -f /tmp/melvin_run.log # Watch live logs"
echo "  ./melvin_monitor brain.m 5  # Monitor brain stats"
echo "  exit                        # Disconnect"
echo ""
echo "Connecting..."
echo ""

# Open interactive SSH session and cd to melvin directory
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" \
    "cd /home/melvin/melvin && exec bash -l"

