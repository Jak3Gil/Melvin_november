#!/bin/bash
# Interactive chat script for Melvin
# Allows you to talk to Melvin and see responses

set -e

BRAIN_FILE="${1:-melvin.m}"

if [ ! -f "$BRAIN_FILE" ]; then
    echo "Error: Brain file '$BRAIN_FILE' not found"
    echo "Run: ./init_melvin_jetson.sh first"
    exit 1
fi

echo "=========================================="
echo "Chatting with Melvin"
echo "=========================================="
echo ""
echo "Type your messages and press Enter"
echo "Melvin will respond based on his graph"
echo "Type 'quit' or 'exit' to stop"
echo ""

# Run melvin in background and interact with it
./melvin "$BRAIN_FILE" 2>&1 &
MELVIN_PID=$!

# Cleanup on exit
trap "kill $MELVIN_PID 2>/dev/null; exit" INT TERM EXIT

# Simple chat loop
while true; do
    echo -n "You: "
    read -r input
    
    if [ "$input" = "quit" ] || [ "$input" = "exit" ]; then
        break
    fi
    
    # Send input to Melvin (he reads from stdin)
    echo "$input"
    
    # Give Melvin time to process
    sleep 0.5
done

echo ""
echo "Goodbye!"
kill $MELVIN_PID 2>/dev/null

