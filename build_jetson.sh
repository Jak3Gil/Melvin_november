#!/bin/bash
# Build script for Jetson Orin AGX
# Optimized for ARM64 architecture with 64GB RAM

set -e

echo "=========================================="
echo "Building Melvin for Jetson Orin AGX"
echo "=========================================="

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "Warning: Not running on ARM64. Cross-compilation may be needed."
fi

# Compiler settings for Jetson
export CC=gcc
export CXX=g++
export CFLAGS="-O3 -Wall -Wextra -std=c11 -fPIC -march=armv8.2-a+fp16+simd -mtune=cortex-a78 -ffast-math"
export LDFLAGS="-shared -fPIC -lm -ldl -lpthread"

# Check for required tools
echo "Checking build tools..."
command -v gcc >/dev/null 2>&1 || { echo "Error: gcc not found. Install: sudo apt install build-essential"; exit 1; }
command -v make >/dev/null 2>&1 || { echo "Error: make not found. Install: sudo apt install build-essential"; exit 1; }

# Build main executable
echo ""
echo "Building main executable..."
gcc $CFLAGS -c melvin.c -o melvin.o
gcc $CFLAGS -o melvin melvin.o -lm -ldl -lpthread

# Build plugins
echo ""
echo "Building plugins..."
PLUGIN_DIR="plugins"
mkdir -p "$PLUGIN_DIR"

for plugin in "$PLUGIN_DIR"/*.c; do
    if [ -f "$plugin" ]; then
        plugin_name=$(basename "$plugin" .c)
        plugin_so="$PLUGIN_DIR/${plugin_name}.so"
        echo "  Building: $plugin_name"
        gcc $CFLAGS $LDFLAGS -I. -undefined dynamic_lookup -o "$plugin_so" "$plugin"
    fi
done

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo "Main executable: ./melvin"
echo "Plugins: plugins/*.so"
echo ""
echo "To run: ./melvin [melvin.m]"
echo ""

