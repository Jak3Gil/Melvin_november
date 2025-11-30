#!/bin/bash

# 20-Minute Melvin Test Runner
# Compiles and runs the 20-minute sanity test

set -e

echo "=========================================="
echo "MELVIN 20-MINUTE TEST RUNNER"
echo "=========================================="
echo ""

# Check if melvin.c exists
if [ ! -f "melvin.c" ]; then
    echo "ERROR: melvin.c not found in current directory"
    exit 1
fi

# Compile the test program
echo "Compiling test_run_20min.c..."
gcc -o test_run_20min test_run_20min.c -lm -std=c11 -Wall -Wextra

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Check if test file already exists
if [ -f "test_20min.m" ]; then
    echo "Warning: test_20min.m already exists"
    read -p "Delete and create fresh? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f test_20min.m
        echo "✓ Removed old test file"
    else
        echo "Keeping existing file"
    fi
    echo ""
fi

# Run the test
echo "Starting 20-minute test..."
echo "(This will take exactly 20 minutes)"
echo ""
echo "Press Ctrl+C to stop early"
echo ""

./test_run_20min

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ TEST COMPLETED SUCCESSFULLY"
else
    echo "✗ TEST FAILED OR WAS INTERRUPTED"
fi
echo "=========================================="

exit $EXIT_CODE

