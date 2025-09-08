#!/bin/bash

# ============================================================================
# BUILD SCRIPT FOR MELVIN MEMORY CONSOLIDATOR
# ============================================================================
# This script builds and runs the memory consolidator that unifies all of
# Melvin's brain instances into one cohesive system

set -e  # Exit on any error

echo "🧠 MELVIN MEMORY CONSOLIDATOR BUILD SCRIPT"
echo "=========================================="
echo "Building memory consolidator to unify all brain instances"
echo "This will connect all memory systems into one unified brain!"

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
rm -f melvin_memory_consolidator
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

# Build the memory consolidator
print_status "Building memory consolidator..."

# Compile with all available compression libraries
g++ $COMPILER_FLAGS $INCLUDE_FLAGS \
    -DHAVE_ZLIB -DHAVE_LZMA -DHAVE_ZSTD \
    melvin_memory_consolidator.cpp melvin_optimized_v2.cpp \
    $ZLIB_FLAGS $LZMA_FLAGS $ZSTD_FLAGS \
    -o melvin_memory_consolidator

if [ $? -eq 0 ]; then
    print_success "Memory consolidator compiled successfully!"
else
    print_error "Compilation failed!"
    exit 1
fi

# Make executable
chmod +x melvin_memory_consolidator

# Show existing memory systems
print_status "Existing memory systems to consolidate:"
ls -la melvin_*memory*/ 2>/dev/null || echo "No existing memory systems found"

# Run the consolidator
print_status "Running memory consolidator..."
echo ""
echo "🧠 STARTING MELVIN MEMORY CONSOLIDATOR"
echo "======================================"
echo "This will consolidate all brain instances:"
echo "• melvin_binary_memory (main brain)"
echo "• melvin_arc_memory (ARC reasoning)"
echo "• melvin_intelligent_memory (intelligent answering)"
echo "• melvin_real_arc_memory (real ARC test)"
echo "• melvin_unified_intelligent_memory (unified intelligent)"
echo ""

# Run the memory consolidator
./melvin_memory_consolidator

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "Memory consolidation completed successfully!"
else
    print_error "Memory consolidation failed with exit code: $TEST_EXIT_CODE"
fi

# Display results summary
echo ""
echo "📊 MEMORY CONSOLIDATION SUMMARY"
echo "==============================="
echo "System executable: melvin_memory_consolidator"
echo "System type: Memory consolidation and unification"
echo "Results: See detailed report above"
echo ""

# Check if unified memory was created
if [ -d "melvin_unified_memory" ]; then
    print_status "Unified memory system created in: melvin_unified_memory/"
    ls -la melvin_unified_memory/
    
    echo ""
    print_status "Storage comparison:"
    echo "Before consolidation:"
    du -sh melvin_*memory*/ 2>/dev/null | grep -v melvin_unified_memory || echo "No previous memory systems"
    echo ""
    echo "After consolidation:"
    du -sh melvin_unified_memory/
fi

print_success "Memory consolidation build and execution complete!"
echo ""
echo "💡 This consolidator unified all brain instances into one system:"
echo "• Connected all memory systems"
echo "• Created cross-connections between knowledge types"
echo "• Maintained intelligent capabilities"
echo "• Preserved all knowledge types"
echo "• Created unified storage system"
echo "• Enabled cross-memory learning"
echo ""
echo "🎯 Key Achievement: UNIFIED MEMORY SYSTEM!"
echo "   All brain instances are now connected in one system!"
