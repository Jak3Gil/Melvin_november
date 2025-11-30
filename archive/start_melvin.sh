#!/bin/bash
# Start Melvin - runs melvin.m continuously

BRAIN_FILE="${1:-melvin.m}"

if [ ! -f "$BRAIN_FILE" ]; then
    echo "ERROR: Brain file $BRAIN_FILE not found!"
    echo "Run: ./init_melvin_simple to create it first"
    exit 1
fi

echo "=========================================="
echo "Starting Melvin"
echo "=========================================="
echo "Brain file: $BRAIN_FILE"
echo "Press Ctrl+C to stop"
echo ""

# Check if melvin is already running
if [ -f melvin.pid ]; then
    OLD_PID=$(cat melvin.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Melvin is already running (PID: $OLD_PID)"
        echo "Stop it first or remove melvin.pid"
        exit 1
    fi
    rm -f melvin.pid
fi

# Compile if needed
if [ ! -f melvin ] || [ melvin.c -nt melvin ]; then
    echo "Compiling melvin..."
    gcc -o melvin melvin.c -lm -ldl 2>&1 | grep -E "(error|warning)" || echo "✓ Compiled"
fi

# Start melvin in background
echo "Starting melvin process..."
./melvin "$BRAIN_FILE" > melvin.log 2>&1 &
MELVIN_PID=$!
echo $MELVIN_PID > melvin.pid

echo "Melvin started (PID: $MELVIN_PID)"
echo "Log file: melvin.log"
echo ""
echo "To monitor: ./monitor_melvin.sh"
echo "To stop: kill $MELVIN_PID or ./stop_melvin.sh"
echo ""

# Wait for process
wait $MELVIN_PID
rm -f melvin.pid
echo "Melvin stopped"

