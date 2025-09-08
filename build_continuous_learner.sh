#!/bin/bash

# ============================================================================
# BUILD SCRIPT FOR MELVIN CONTINUOUS LEARNER
# ============================================================================
# This script builds the continuous learning system that runs Melvin
# continuously, searches for knowledge gaps, and uses Ollama to fill them

echo "🧠 Building Melvin Continuous Learner..."
echo "========================================"

# Check if required libraries are installed
echo "📋 Checking dependencies..."

# Check for compression libraries
if ! pkg-config --exists zlib; then
    echo "❌ zlib not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install zlib
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install -y zlib1g-dev
    fi
fi

if ! pkg-config --exists liblzma; then
    echo "❌ liblzma not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install xz
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install -y liblzma-dev
    fi
fi

if ! pkg-config --exists libzstd; then
    echo "❌ libzstd not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install zstd
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install -y libzstd-dev
    fi
fi

echo "✅ Dependencies checked"

# Compile with optimizations
echo "🔨 Compiling Melvin Continuous Learner..."

g++ -std=c++17 -O3 -march=native -ffast-math \
    -Wall -Wextra -Wpedantic \
    -DHAVE_ZLIB -DHAVE_LZMA -DHAVE_ZSTD \
    -I/usr/local/include \
    -L/usr/local/lib \
    melvin_continuous_learner.cpp \
    melvin_optimized_v2.cpp \
    -lz -llzma -lzstd \
    -pthread \
    -o melvin_continuous_learner

if [ $? -eq 0 ]; then
    echo "✅ Melvin Continuous Learner compiled successfully!"
    echo ""
    echo "🚀 To run Melvin continuously:"
    echo "   ./melvin_continuous_learner"
    echo ""
    echo "📋 Features:"
    echo "   • Continuous learning with knowledge gap detection"
    echo "   • Recall Track and Exploration Track reasoning"
    echo "   • Self-regulator system for input filtering"
    echo "   • Ollama integration for knowledge filling"
    echo "   • Automatic saves to global repository every 2 minutes"
    echo "   • Integrated conclusions with confidence scoring"
    echo ""
    echo "🎯 Melvin will:"
    echo "   • Run continuously and process inputs"
    echo "   • Search for holes in his knowledge"
    echo "   • Use Ollama to help fill knowledge gaps"
    echo "   • Generate reasoning tracks for each input"
    echo "   • Save progress to global repository regularly"
else
    echo "❌ Compilation failed!"
    echo "Please check the error messages above and fix any issues."
    exit 1
fi

echo ""
echo "🧠 Melvin Continuous Learner is ready to learn continuously!"
