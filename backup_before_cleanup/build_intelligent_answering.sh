#!/bin/bash

# ============================================================================
# BUILD SCRIPT FOR MELVIN INTELLIGENT ANSWERING SYSTEM
# ============================================================================
# This script builds and runs the intelligent answering system that uses
# Melvin's existing brain architecture for connection path traversal and
# dynamic node creation

set -e  # Exit on any error

echo "🧠 MELVIN INTELLIGENT ANSWERING BUILD SCRIPT"
echo "==========================================="
echo "Building intelligent answering system using Melvin's brain architecture"
echo "This demonstrates connection path traversal and dynamic node creation"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "melvin_optimized_v2.h" ]; then
    print_error "melvin_optimized_v2.h not found. Please run from project root."
    exit 1
fi

# Clean previous builds
print_status "Cleaning previous builds..."
rm -f melvin_intelligent_answering
rm -f *.o

# Compiler flags for optimization
COMPILER_FLAGS="-std=c++17 -O3 -march=native -ffast-math -Wall -Wextra"
INCLUDE_FLAGS="-I. -I/usr/local/include -I/opt/homebrew/include"
LINK_FLAGS="-L/usr/local/lib -L/opt/homebrew/lib"

# Try to find compression libraries
ZLIB_FLAGS=""
LZMA_FLAGS=""
ZSTD_FLAGS=""

# Zlib
if pkg-config --exists zlib; then
    ZLIB_FLAGS=$(pkg-config --cflags --libs zlib)
elif [ -f "/usr/local/include/zlib.h" ] || [ -f "/opt/homebrew/include/zlib.h" ]; then
    ZLIB_FLAGS="-lz"
fi

# LZMA
if pkg-config --exists liblzma; then
    LZMA_FLAGS=$(pkg-config --cflags --libs liblzma)
elif [ -f "/usr/local/include/lzma.h" ] || [ -f "/opt/homebrew/include/lzma.h" ]; then
    LZMA_FLAGS="-llzma"
fi

# ZSTD
if pkg-config --exists libzstd; then
    ZSTD_FLAGS=$(pkg-config --cflags --libs libzstd)
elif [ -f "/usr/local/include/zstd.h" ] || [ -f "/opt/homebrew/include/zstd.h" ]; then
    ZSTD_FLAGS="-lzstd"
fi

# Build the intelligent answering system
print_status "Building intelligent answering system..."

# Compile with all available compression libraries
g++ $COMPILER_FLAGS $INCLUDE_FLAGS \
    -DHAVE_ZLIB -DHAVE_LZMA -DHAVE_ZSTD \
    melvin_intelligent_answering.cpp melvin_optimized_v2.cpp \
    $ZLIB_FLAGS $LZMA_FLAGS $ZSTD_FLAGS \
    -o melvin_intelligent_answering

if [ $? -eq 0 ]; then
    print_success "Intelligent answering system compiled successfully!"
else
    print_error "Compilation failed!"
    exit 1
fi

# Make executable
chmod +x melvin_intelligent_answering

# Run the test
print_status "Running intelligent answering system..."
echo ""
echo "🧠 STARTING MELVIN INTELLIGENT ANSWERING SYSTEM"
echo "==============================================="
echo "This system demonstrates:"
echo "• Connection path traversal to find relevant knowledge"
echo "• Answer synthesis from partial knowledge"
echo "• Dynamic node creation for future questions"
echo "• Intelligent reasoning when no perfect match exists"
echo ""

# Run the intelligent answering system
./melvin_intelligent_answering

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "Intelligent answering system completed successfully!"
else
    print_error "Intelligent answering system failed with exit code: $TEST_EXIT_CODE"
fi

# Display results summary
echo ""
echo "📊 INTELLIGENT ANSWERING SYSTEM SUMMARY"
echo "======================================="
echo "System executable: melvin_intelligent_answering"
echo "System type: Connection path traversal and dynamic node creation"
echo "Results: See detailed report above"
echo ""

# Check if memory files were created
if [ -d "melvin_intelligent_memory" ]; then
    print_status "Intelligent answering memory files created in: melvin_intelligent_memory/"
    ls -la melvin_intelligent_memory/
fi

print_success "Intelligent answering system build and execution complete!"
echo ""
echo "💡 This system demonstrated Melvin's ability to:"
echo "• Answer questions he doesn't have perfect answers for"
echo "• Use connection paths to find relevant knowledge"
echo "• Generalize from existing nodes"
echo "• Create new nodes dynamically"
echo "• Synthesize intelligent responses"
echo ""
echo "🎯 Key Insight: This uses Melvin's EXISTING brain architecture!"
echo "   No separate reasoning engine needed - just intelligent connection traversal!"
