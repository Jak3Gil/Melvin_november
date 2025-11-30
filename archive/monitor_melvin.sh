#!/bin/bash
# Monitor Melvin brain file - shows live stats

BRAIN_FILE="${1:-melvin.m}"
REFRESH_RATE="${2:-0.5}"  # seconds

if [ ! -f "$BRAIN_FILE" ]; then
    echo "ERROR: Brain file $BRAIN_FILE not found!"
    exit 1
fi

# Compile monitor if needed
if [ ! -f monitor_melvin ] || [ monitor_melvin.c -nt monitor_melvin ]; then
    echo "Compiling monitor..."
    gcc -o monitor_melvin monitor_melvin.c -lm 2>&1 | grep -E "(error|warning)" || true
fi

echo "=========================================="
echo "Melvin Monitor"
echo "=========================================="
echo "Brain file: $BRAIN_FILE"
echo "Refresh rate: ${REFRESH_RATE}s"
echo "Press Ctrl+C to stop"
echo ""

./monitor_melvin "$BRAIN_FILE" "$REFRESH_RATE"

