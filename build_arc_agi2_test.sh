#!/bin/bash

# ============================================================================
# BUILD SCRIPT FOR MELVIN ARC AGI-2 TEST
# ============================================================================
# This script builds and runs the comprehensive ARC AGI-2 test suite
# to evaluate Melvin's artificial general intelligence capabilities

set -e  # Exit on any error

echo "🧠 MELVIN ARC AGI-2 TEST BUILD SCRIPT"
echo "===================================="
echo "Building comprehensive AGI evaluation test suite..."

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
rm -f melvin_arc_agi2_test
rm -f *.o

# Check for required libraries
print_status "Checking for required libraries..."

# Check for compression libraries
MISSING_LIBS=""

if ! pkg-config --exists zlib; then
    MISSING_LIBS="$MISSING_LIBS zlib"
fi

if ! pkg-config --exists liblzma; then
    MISSING_LIBS="$MISSING_LIBS liblzma"
fi

if ! pkg-config --exists libzstd; then
    MISSING_LIBS="$MISSING_LIBS libzstd"
fi

if [ ! -z "$MISSING_LIBS" ]; then
    print_warning "Missing compression libraries: $MISSING_LIBS"
    print_status "Installing missing libraries..."
    
    # Detect OS and install accordingly
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install zlib xz zstd
        else
            print_error "Homebrew not found. Please install: brew install zlib xz zstd"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y zlib1g-dev liblzma-dev libzstd-dev
        elif command -v yum &> /dev/null; then
            sudo yum install -y zlib-devel xz-devel libzstd-devel
        else
            print_error "Package manager not found. Please install: zlib-devel xz-devel libzstd-devel"
            exit 1
        fi
    else
        print_error "Unsupported OS. Please install compression libraries manually."
        exit 1
    fi
fi

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

# Build the ARC AGI-2 test
print_status "Building ARC AGI-2 test..."

# Compile with all available compression libraries
g++ $COMPILER_FLAGS $INCLUDE_FLAGS \
    -DHAVE_ZLIB -DHAVE_LZMA -DHAVE_ZSTD \
    melvin_arc_agi2_test.cpp melvin_optimized_v2.cpp \
    $ZLIB_FLAGS $LZMA_FLAGS $ZSTD_FLAGS \
    -o melvin_arc_agi2_test

if [ $? -eq 0 ]; then
    print_success "ARC AGI-2 test compiled successfully!"
else
    print_error "Compilation failed!"
    exit 1
fi

# Make executable
chmod +x melvin_arc_agi2_test

# Run the test
print_status "Running ARC AGI-2 comprehensive test..."
echo ""
echo "🧠 STARTING MELVIN ARC AGI-2 EVALUATION"
echo "========================================"
echo "This test evaluates Melvin's AGI capabilities including:"
echo "• Pattern recognition and abstraction"
echo "• Multi-step reasoning and logic"
echo "• Cross-domain knowledge transfer"
echo "• Meta-cognitive abilities"
echo ""

# Run the test with timeout (30 minutes max)
timeout 1800 ./melvin_arc_agi2_test

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "ARC AGI-2 test completed successfully!"
elif [ $TEST_EXIT_CODE -eq 124 ]; then
    print_warning "ARC AGI-2 test timed out after 30 minutes"
else
    print_error "ARC AGI-2 test failed with exit code: $TEST_EXIT_CODE"
fi

# Display results summary
echo ""
echo "📊 ARC AGI-2 TEST SUMMARY"
echo "========================"
echo "Test executable: melvin_arc_agi2_test"
echo "Test duration: Check output above for timing"
echo "Results: See detailed report above"
echo ""

# Check if memory files were created
if [ -d "melvin_arc_memory" ]; then
    print_status "ARC test memory files created in: melvin_arc_memory/"
    ls -la melvin_arc_memory/
fi

print_success "ARC AGI-2 test build and execution complete!"
echo ""
echo "💡 Next steps:"
echo "• Review the test results above"
echo "• Check Melvin's performance in different AGI categories"
echo "• Use results to identify areas for improvement"
echo "• Run additional tests to validate improvements"
