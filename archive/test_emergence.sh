#!/bin/bash

# Melvin Emergence Test Script

set -e

echo "=========================================="
echo "MELVIN EMERGENCE TEST"
echo "=========================================="
echo ""

# Build the system
echo "[1/3] Building Melvin..."
make -f Makefile.melvin clean
make -f Makefile.melvin

# Build test program
echo "[2/3] Building emergence test..."
clang -Wall -Wextra -O2 -g -std=c11 -I. -o test_emergence test_emergence.c melvin_file.c melvin_runtime.c -lm

# Check if build succeeded
if [ ! -f test_emergence ]; then
    echo "ERROR: Failed to build test program"
    exit 1
fi

# Run test
echo "[3/3] Running emergence test..."
echo ""
./test_emergence -t 5000 -r 500

echo ""
echo "=========================================="
echo "Test complete! Check melvin.m brain file."
echo "=========================================="

