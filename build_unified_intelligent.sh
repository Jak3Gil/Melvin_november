#!/bin/bash

# ============================================================================
# BUILD SCRIPT FOR MELVIN UNIFIED INTELLIGENT TEST
# ============================================================================
# This script builds and runs the unified intelligent test that demonstrates
# Melvin's brain with integrated intelligent connection traversal and
# dynamic node creation capabilities

set -e  # Exit on any error

echo "🧠 MELVIN UNIFIED INTELLIGENT BUILD SCRIPT"
echo "=========================================="
echo "Building unified intelligent test with integrated capabilities"
echo "Every interaction now includes intelligent connection traversal!"

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
rm -f melvin_unified_intelligent_test
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

# Build the unified intelligent test
print_status "Building unified intelligent test..."

# Compile with all available compression libraries
g++ $COMPILER_FLAGS $INCLUDE_FLAGS \
    -DHAVE_ZLIB -DHAVE_LZMA -DHAVE_ZSTD \
    melvin_unified_intelligent_test.cpp melvin_optimized_v2.cpp \
    $ZLIB_FLAGS $LZMA_FLAGS $ZSTD_FLAGS \
    -o melvin_unified_intelligent_test

if [ $? -eq 0 ]; then
    print_success "Unified intelligent test compiled successfully!"
else
    print_error "Compilation failed!"
    exit 1
fi

# Make executable
chmod +x melvin_unified_intelligent_test

# Run the test
print_status "Running unified intelligent test..."
echo ""
echo "🧠 STARTING MELVIN UNIFIED INTELLIGENT TEST"
echo "==========================================="
echo "This test demonstrates:"
echo "• Unified brain architecture with intelligent capabilities"
echo "• Automatic connection path traversal"
echo "• Dynamic node creation for new knowledge"
echo "• Intelligent answer synthesis from partial knowledge"
echo "• Keyword extraction and relevant node discovery"
echo "• Hebbian learning with intelligent processing"
echo ""

# Run the unified intelligent test
./melvin_unified_intelligent_test

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "Unified intelligent test completed successfully!"
else
    print_error "Unified intelligent test failed with exit code: $TEST_EXIT_CODE"
fi

# Display results summary
echo ""
echo "📊 UNIFIED INTELLIGENT TEST SUMMARY"
echo "==================================="
echo "System executable: melvin_unified_intelligent_test"
echo "System type: Unified brain with integrated intelligent capabilities"
echo "Results: See detailed report above"
echo ""

# Check if memory files were created
if [ -d "melvin_unified_intelligent_memory" ]; then
    print_status "Unified intelligent memory files created in: melvin_unified_intelligent_memory/"
    ls -la melvin_unified_intelligent_memory/
fi

print_success "Unified intelligent test build and execution complete!"
echo ""
echo "💡 This test demonstrated Melvin's unified brain with:"
echo "• Integrated intelligent connection traversal"
echo "• Automatic dynamic node creation"
echo "• Intelligent answer synthesis"
echo "• Keyword extraction and relevant node discovery"
echo "• Hebbian learning with intelligent processing"
echo "• Binary storage with intelligent capabilities"
echo ""
echo "🎯 Key Achievement: UNIFIED BRAIN ARCHITECTURE!"
echo "   Every interaction now includes intelligent capabilities!"
echo "   No separate systems needed - everything is integrated!"
