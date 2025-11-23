#!/bin/bash
# Run script for Jetson Orin AGX
# Optimized runtime settings for 64GB RAM

set -e

echo "=========================================="
echo "Starting Melvin on Jetson Orin AGX"
echo "=========================================="

# Check if melvin exists
if [ ! -f "./melvin" ]; then
    echo "Error: melvin executable not found"
    echo "Run: ./build_jetson.sh first"
    exit 1
fi

# Jetson-specific optimizations
export OMP_NUM_THREADS=8  # Use 8 CPU cores
export MALLOC_ARENA_MAX=4  # Reduce memory fragmentation

# Set memory limits (64GB available - generous limits)
ulimit -v 53687091200  # 50GB virtual memory limit
ulimit -m 53687091200  # 50GB physical memory limit

# Check for melvin.m
BRAIN_FILE="${1:-melvin.m}"
if [ ! -f "$BRAIN_FILE" ]; then
    echo "Warning: Brain file '$BRAIN_FILE' not found"
    echo "Melvin will try to create it, or you may need to initialize it first"
fi

# Run with Jetson optimizations
echo "Running Melvin with:"
echo "  Brain file: $BRAIN_FILE"
echo "  CPU threads: $OMP_NUM_THREADS"
echo "  Memory limit: 50GB"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run melvin
./melvin "$BRAIN_FILE"

