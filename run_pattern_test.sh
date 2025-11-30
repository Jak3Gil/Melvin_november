#!/bin/bash

# Pattern Stress-Test + Reward Loop Runner
# Tests pattern learning with minimal reward injection

set -e

echo "=========================================="
echo "MELVIN PATTERN + REWARD TEST RUNNER"
echo "=========================================="
echo ""

# Check if melvin.c exists
if [ ! -f "melvin.c" ]; then
    echo "ERROR: melvin.c not found in current directory"
    exit 1
fi

# Compile the test program
echo "Compiling test_pattern_reward.c..."
gcc -o test_pattern_reward test_pattern_reward.c -lm -std=c11 -Wall -Wextra

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Check if baseline file exists
if [ -f "test_20min.m" ]; then
    echo "Found baseline file: test_20min.m"
    echo "Will reuse this stable baseline for the test"
else
    echo "Note: test_20min.m not found - will create new baseline"
fi
echo ""

# Run the test
echo "Starting pattern stress-test with reward loop..."
echo "(This will take exactly 10 minutes)"
echo ""
echo "Press Ctrl+C to stop early"
echo ""

./test_pattern_reward

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

