#!/bin/bash

# Universal Stress Test Runner
# Runs comprehensive production-level stress test

set -e

echo "=========================================="
echo "MELVIN UNIVERSAL STRESS TEST RUNNER"
echo "=========================================="
echo ""

# Check if melvin.c exists
if [ ! -f "melvin.c" ]; then
    echo "ERROR: melvin.c not found in current directory"
    exit 1
fi

# Compile the test program
echo "Compiling test_universal_stress.c..."
gcc -o test_universal_stress test_universal_stress.c -lm -std=c11 -Wall -Wextra -O2

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi

echo "✓ Compilation successful"
echo ""

# Check if test file already exists
if [ -f "test_universal_stress.m" ]; then
    echo "Warning: test_universal_stress.m already exists"
    read -p "Delete and create fresh? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f test_universal_stress.m
        echo "✓ Removed old test file"
    else
        echo "Keeping existing file"
    fi
    echo ""
fi

# Run the test
echo "Starting universal stress test..."
echo "This will test:"
echo "  - File operations & validation"
echo "  - All runtime functions"
echo "  - Edge cases & error conditions"
echo "  - Large-scale stress operations"
echo "  - Production workload simulation"
echo ""
echo "Press Ctrl+C to stop early"
echo ""

./test_universal_stress

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓✓✓ TEST COMPLETED SUCCESSFULLY ✓✓✓"
else
    echo "✗✗✗ TEST FAILED OR WAS INTERRUPTED ✗✗✗"
fi
echo "=========================================="

exit $EXIT_CODE

