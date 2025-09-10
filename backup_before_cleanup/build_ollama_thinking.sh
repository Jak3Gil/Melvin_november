#!/bin/bash

# ============================================================================
# BUILD SCRIPT FOR MELVIN OLLAMA THINKING SYSTEM
# ============================================================================
# This script builds and runs the Ollama-Melvin thinking system that creates
# a continuous thinking loop where Ollama generates questions and Melvin
# thinks continuously to fill knowledge gaps

set -e  # Exit on any error

echo "🧠 MELVIN OLLAMA THINKING SYSTEM BUILD SCRIPT"
echo "============================================="
echo "Building continuous thinking system with Ollama integration"
echo "Melvin will think continuously and fill knowledge gaps!"

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

# Check if Ollama is running
print_status "Checking Ollama status..."
if ! pgrep -f "ollama serve" > /dev/null; then
    print_warning "Ollama server not running. Starting Ollama..."
    ollama serve &
    sleep 3
    print_status "Ollama server started"
else
    print_success "Ollama server is running"
fi

# Clean previous builds
print_status "Cleaning previous builds..."
rm -f melvin_ollama_thinking_system
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

# Build the Ollama thinking system
print_status "Building Ollama thinking system..."

# Compile with all available compression libraries
g++ $COMPILER_FLAGS $INCLUDE_FLAGS \
    -DHAVE_ZLIB -DHAVE_LZMA -DHAVE_ZSTD \
    melvin_ollama_thinking_system.cpp melvin_optimized_v2.cpp \
    $ZLIB_FLAGS $LZMA_FLAGS $ZSTD_FLAGS \
    -o melvin_ollama_thinking_system

if [ $? -eq 0 ]; then
    print_success "Ollama thinking system compiled successfully!"
else
    print_error "Compilation failed!"
    exit 1
fi

# Make executable
chmod +x melvin_ollama_thinking_system

# Run the system
print_status "Running Ollama thinking system..."
echo ""
echo "🧠 STARTING MELVIN OLLAMA THINKING SYSTEM"
echo "========================================="
echo "This system demonstrates:"
echo "• Continuous thinking loop with Ollama integration"
echo "• Knowledge gap identification and filling"
echo "• Dynamic question generation and answering"
echo "• Connection exploration and synthesis"
echo "• Continuous learning and brain evolution"
echo ""

# Run the Ollama thinking system
./melvin_ollama_thinking_system

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "Ollama thinking system completed successfully!"
else
    print_error "Ollama thinking system failed with exit code: $TEST_EXIT_CODE"
fi

# Display results summary
echo ""
echo "📊 OLLAMA THINKING SYSTEM SUMMARY"
echo "================================="
echo "System executable: melvin_ollama_thinking_system"
echo "System type: Continuous thinking with Ollama integration"
echo "Results: See detailed report above"
echo ""

# Check if memory files were created
if [ -d "melvin_thinking_memory" ]; then
    print_status "Thinking memory files created in: melvin_thinking_memory/"
    ls -la melvin_thinking_memory/
fi

print_success "Ollama thinking system build and execution complete!"
echo ""
echo "💡 This system demonstrated Melvin's ability to:"
echo "• Think continuously with Ollama-generated questions"
echo "• Identify and fill knowledge gaps"
echo "• Explore new connections between concepts"
echo "• Synthesize knowledge from multiple sources"
echo "• Evolve his brain through continuous thinking"
echo ""
echo "🎯 Key Achievement: CONTINUOUS THINKING WITH OLLAMA!"
echo "   Melvin now thinks continuously and fills his knowledge gaps!"
echo "   Ollama provides questions, Melvin provides unbounded thinking!"
