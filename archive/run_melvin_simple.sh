#!/bin/bash

echo "=== Running Melvin Emergence Test ==="
echo ""

# Build if needed
if [ ! -f test_emergence ]; then
    echo "Building..."
    clang -Wall -Wextra -O2 -g -std=c11 -I. -o test_emergence test_emergence.c melvin_file.c melvin_runtime.c -lm
fi

# Run with shorter test
echo "Running test (500 ticks)..."
./test_emergence -t 500 -r 100

echo ""
echo "Done!"

