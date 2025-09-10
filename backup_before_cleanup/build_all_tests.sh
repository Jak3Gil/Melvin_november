#!/bin/bash

# ============================================================================
# BUILD ALL MELVIN TESTS SCRIPT
# ============================================================================
# This script builds all the test files we created during our testing journey
# Note: Most of these tests are essentially fake due to hardcoded answers

set -e  # Exit on any error

echo "🧠 MELVIN ALL TESTS BUILD SCRIPT"
echo "==============================="
echo "Building all test files from our testing journey"
echo "WARNING: Most tests use hardcoded answers, not genuine reasoning!"

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
rm -f melvin_real_arc_test
rm -f melvin_genuine_brain_test
rm -f melvin_truly_genuine_test
rm -f melvin_pure_brain_test
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

# Build all test files
print_status "Building all test files..."

# 1. Simulated ARC AGI-2 Test (fake - uses random numbers)
print_status "Building simulated ARC AGI-2 test..."
g++ $COMPILER_FLAGS $INCLUDE_FLAGS \
    -DHAVE_ZLIB -DHAVE_LZMA -DHAVE_ZSTD \
    melvin_arc_agi2_test.cpp melvin_optimized_v2.cpp \
    $ZLIB_FLAGS $LZMA_FLAGS $ZSTD_FLAGS \
    -o melvin_arc_agi2_test

if [ $? -eq 0 ]; then
    print_success "Simulated ARC AGI-2 test compiled successfully!"
else
    print_error "Simulated ARC AGI-2 test compilation failed!"
fi

# 2. "Real" ARC AGI-2 Test (fake - uses hardcoded answers)
print_status "Building 'real' ARC AGI-2 test..."
g++ $COMPILER_FLAGS $INCLUDE_FLAGS \
    -DHAVE_ZLIB -DHAVE_LZMA -DHAVE_ZSTD \
    melvin_real_arc_test.cpp melvin_optimized_v2.cpp \
    $ZLIB_FLAGS $LZMA_FLAGS $ZSTD_FLAGS \
    -o melvin_real_arc_test

if [ $? -eq 0 ]; then
    print_success "'Real' ARC AGI-2 test compiled successfully!"
else
    print_error "'Real' ARC AGI-2 test compilation failed!"
fi

# 3. "Genuine" Brain Test (fake - uses hardcoded answers)
print_status "Building 'genuine' brain test..."
g++ $COMPILER_FLAGS $INCLUDE_FLAGS \
    -DHAVE_ZLIB -DHAVE_LZMA -DHAVE_ZSTD \
    melvin_genuine_brain_test.cpp melvin_optimized_v2.cpp \
    $ZLIB_FLAGS $LZMA_FLAGS $ZSTD_FLAGS \
    -o melvin_genuine_brain_test

if [ $? -eq 0 ]; then
    print_success "'Genuine' brain test compiled successfully!"
else
    print_error "'Genuine' brain test compilation failed!"
fi

# 4. "Truly Genuine" Test (fake - uses hardcoded answers)
print_status "Building 'truly genuine' test..."
g++ $COMPILER_FLAGS $INCLUDE_FLAGS \
    -DHAVE_ZLIB -DHAVE_LZMA -DHAVE_ZSTD \
    melvin_truly_genuine_test.cpp melvin_optimized_v2.cpp \
    $ZLIB_FLAGS $LZMA_FLAGS $ZSTD_FLAGS \
    -o melvin_truly_genuine_test

if [ $? -eq 0 ]; then
    print_success "'Truly genuine' test compiled successfully!"
else
    print_error "'Truly genuine' test compilation failed!"
fi

# 5. "Pure Brain" Test (fake - uses hardcoded answers)
print_status "Building 'pure brain' test..."
g++ $COMPILER_FLAGS $INCLUDE_FLAGS \
    -DHAVE_ZLIB -DHAVE_LZMA -DHAVE_ZSTD \
    melvin_pure_brain_test.cpp melvin_optimized_v2.cpp \
    $ZLIB_FLAGS $LZMA_FLAGS $ZSTD_FLAGS \
    -o melvin_pure_brain_test

if [ $? -eq 0 ]; then
    print_success "'Pure brain' test compiled successfully!"
else
    print_error "'Pure brain' test compilation failed!"
fi

# Make all executables
chmod +x melvin_arc_agi2_test
chmod +x melvin_real_arc_test
chmod +x melvin_genuine_brain_test
chmod +x melvin_truly_genuine_test
chmod +x melvin_pure_brain_test

# Display summary
echo ""
echo "📊 BUILD SUMMARY"
echo "================"
echo "All test files have been built:"
echo "• melvin_arc_agi2_test (simulated - uses random numbers)"
echo "• melvin_real_arc_test (fake - uses hardcoded answers)"
echo "• melvin_genuine_brain_test (fake - uses hardcoded answers)"
echo "• melvin_truly_genuine_test (fake - uses hardcoded answers)"
echo "• melvin_pure_brain_test (fake - uses hardcoded answers)"
echo ""
echo "⚠️  WARNING: All tests use hardcoded answers, not genuine reasoning!"
echo "   See HONEST_MELVIN_ASSESSMENT.md for details."
echo ""
echo "💡 To run a test:"
echo "   ./melvin_arc_agi2_test"
echo "   ./melvin_real_arc_test"
echo "   ./melvin_genuine_brain_test"
echo "   ./melvin_truly_genuine_test"
echo "   ./melvin_pure_brain_test"
echo ""
echo "📚 Documentation:"
echo "   • HONEST_MELVIN_ASSESSMENT.md - Honest assessment of capabilities"
echo "   • TESTING_JOURNEY.md - Complete testing journey documentation"
echo "   • ARC_AGI2_BENCHMARK_COMPARISON.md - Comparison with other AI systems"
echo ""

print_success "All tests built successfully!"
echo ""
echo "🎯 Next Steps:"
echo "1. Read HONEST_MELVIN_ASSESSMENT.md for honest capabilities assessment"
echo "2. Read TESTING_JOURNEY.md for complete testing journey"
echo "3. Implement actual reasoning capabilities in Melvin's brain"
echo "4. Create genuine tests that evaluate real reasoning"
echo ""
echo "Remember: The current tests are essentially fake due to hardcoded answers!"
echo "We need to build actual reasoning capabilities first."
