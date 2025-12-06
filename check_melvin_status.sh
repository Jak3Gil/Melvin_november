#!/bin/bash
# check_melvin_status.sh - Check Melvin status on Jetson

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"

echo "=========================================="
echo "Melvin Status on Jetson"
echo "=========================================="
echo ""

# Check running processes
echo "Running Processes:"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" "ps aux | grep melvin_run_continuous | grep -v grep || echo '  No melvin processes running'"
echo ""

# Check latest log output
echo "Latest Activity (from /tmp/melvin_run.log):"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" "tail -10 /tmp/melvin_run.log 2>/dev/null || echo '  No log file found'"
echo ""

# Check brain file
echo "Brain File Status:"
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_USER@$JETSON_IP" "ls -lh /mnt/melvin_ssd/melvin/brain.m ~/melvin/brain.m 2>/dev/null | head -2 || echo '  Brain file not found'"
echo ""








