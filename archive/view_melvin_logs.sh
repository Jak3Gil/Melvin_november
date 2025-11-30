#!/bin/bash
# View Melvin's logs in real-time (headless mode)

set -e

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${JETSON_USER}@${JETSON_IP}"

echo "=========================================="
echo "Melvin Log Viewer (Real-time)"
echo "=========================================="
echo ""
echo "Streaming Melvin logs from Jetson..."
echo "Press Ctrl+C to stop"
echo ""

sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "tail -f /home/melvin/melvin_system/melvin.log 2>/dev/null || journalctl -u melvin.service -f --no-pager"

