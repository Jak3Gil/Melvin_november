#!/bin/bash

# ============================================================================
# BUILD SCRIPT FOR MELVIN WORK ANALYZER
# ============================================================================
# This script builds and runs the Melvin work analyzer to understand:
# 1. What knowledge Melvin has accumulated
# 2. How his connections have formed
# 3. What patterns he has learned
# 4. How his synthesis has improved
# 5. What gaps still exist in his knowledge

set -e  # Exit on any error

echo "🔍 MELVIN WORK ANALYZER BUILD SCRIPT"
echo "===================================="
echo "Building work analyzer to understand Melvin's progress"
echo "Analyzing knowledge accumulation, connections, and synthesis!"

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

# Create build directory
print_status "Creating build directory..."
mkdir -p build

# Compile the work analyzer
print_status "Compiling Melvin work analyzer..."
g++ -std=c++17 -O3 -Wall -Wextra \
    -I. \
    melvin_work_analyzer.cpp \
    melvin_optimized_v2.cpp \
    -o build/melvin_work_analyzer \
    -pthread

if [ $? -eq 0 ]; then
    print_success "Work analyzer compiled successfully!"
else
    print_error "Compilation failed!"
    exit 1
fi

# Check if Melvin's memory exists
if [ -d "melvin_unified_intelligent_memory" ]; then
    print_success "Found Melvin's memory directory!"
    print_status "Memory contents:"
    ls -la melvin_unified_intelligent_memory/
else
    print_warning "Melvin's memory directory not found. He may not have run yet."
    print_status "Available memory directories:"
    ls -la | grep memory
fi

# Run the work analyzer
print_status "Running Melvin work analyzer..."
echo ""
echo "🔍 ANALYZING MELVIN'S WORK..."
echo "============================="
echo ""

./build/melvin_work_analyzer

print_success "Work analysis complete!"
