#!/bin/bash

# EXEC Pattern Actor Test Runner
# Tests machine code execution + learning

set -e

echo "=========================================="
echo "EXEC PATTERN ACTOR TEST RUNNER"
echo "=========================================="
echo ""

# Check if melvin.c exists
if [ ! -f "melvin.c" ]; then
    echo "ERROR: melvin.c not found in current directory"
    exit 1
fi

# Compile the test program
echo "Compiling test_exec_pattern_actor.c..."
gcc -o test_exec_pattern_actor test_exec_pattern_actor.c -lm -std=c11 -Wall -Wextra

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Run the test
echo "Starting EXEC pattern actor test..."
echo "This will run 1000 episodes"
echo ""
echo "Press Ctrl+C to stop early"
echo ""

./test_exec_pattern_actor

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ TEST COMPLETED"
else
    echo "✗ TEST FAILED OR WAS INTERRUPTED"
fi
echo "=========================================="

exit $EXIT_CODE

