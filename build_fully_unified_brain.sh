#!/bin/bash

# ============================================================================
# BUILD SCRIPT FOR MELVIN FULLY UNIFIED BRAIN
# ============================================================================
# This script builds and runs the fully unified brain where thinking and
# memory are completely integrated in one system - no separate files!

set -e  # Exit on any error

echo "🧠 MELVIN FULLY UNIFIED BRAIN BUILD SCRIPT"
echo "==========================================="
echo "Building fully unified brain - thinking and memory in one place!"
echo "No separate files or systems - everything is integrated!"

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
if [ ! -f "melvin_fully_unified_brain.h" ]; then
    print_error "melvin_fully_unified_brain.h not found. Please run from project root."
    exit 1
fi

# Clean previous builds
print_status "Cleaning previous builds..."
rm -f test_fully_unified_brain
rm -f *.o

# Compiler flags for optimization
COMPILER_FLAGS="-std=c++17 -O3 -march=native -ffast-math -Wall -Wextra"
INCLUDE_FLAGS="-I. -I/usr/local/include -I/opt/homebrew/include"
LINK_FLAGS="-L/usr/local/lib -L/opt/homebrew/lib"

# Build the fully unified brain test
print_status "Building fully unified brain test..."

# Compile the test
g++ $COMPILER_FLAGS $INCLUDE_FLAGS \
    test_fully_unified_brain.cpp melvin_fully_unified_brain.cpp \
    -o test_fully_unified_brain

if [ $? -eq 0 ]; then
    print_success "Fully unified brain test compiled successfully!"
else
    print_error "Compilation failed!"
    exit 1
fi

# Make executable
chmod +x test_fully_unified_brain

# Run the test
print_status "Running fully unified brain test..."
echo ""
echo "🧠 STARTING MELVIN FULLY UNIFIED BRAIN TEST"
echo "==========================================="
echo "This test demonstrates:"
echo "• Completely unified thinking and memory"
echo "• No separate files or systems"
echo "• Everything happens in one cohesive brain"
echo "• Learning, thinking, and memory are integrated"
echo "• Real-time processing and reasoning"
echo ""

# Run the fully unified brain test
./test_fully_unified_brain

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "Fully unified brain test completed successfully!"
else
    print_error "Fully unified brain test failed with exit code: $TEST_EXIT_CODE"
fi

# Display results summary
echo ""
echo "📊 FULLY UNIFIED BRAIN TEST SUMMARY"
echo "==================================="
echo "System executable: test_fully_unified_brain"
echo "System type: Fully unified thinking and memory"
echo "Results: See detailed report above"
echo ""

print_success "Fully unified brain test build and execution complete!"
echo ""
echo "💡 This test demonstrated a completely unified brain:"
echo "• Thinking and memory are integrated in one system"
echo "• No separate files or systems needed"
echo "• Everything happens in one cohesive brain"
echo "• Learning, thinking, and memory are unified"
echo "• Real-time processing and reasoning"
echo "• Dynamic node creation and connection formation"
echo "• Intelligent answer synthesis"
echo ""
echo "🎯 Key Achievement: FULLY UNIFIED BRAIN ARCHITECTURE!"
echo "   Thinking and memory are completely integrated!"
echo "   No separate systems - everything is one unified brain!"
