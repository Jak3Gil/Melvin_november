#!/bin/bash

# ============================================================================
# MELVIN INSTINCT ENGINE BUILD SCRIPT
# ============================================================================

echo "🧠 Building Melvin Instinct Engine..."
echo "====================================="

# Set compiler flags for optimization
CXX_FLAGS="-std=c++17 -O3 -Wall -Wextra -march=native -ffast-math"

# Create build directory
mkdir -p build

# Compile the instinct engine
echo "📦 Compiling Instinct Engine..."
g++ $CXX_FLAGS -c melvin_instinct_engine.cpp -o build/melvin_instinct_engine.o

if [ $? -eq 0 ]; then
    echo "✅ Instinct Engine compiled successfully"
else
    echo "❌ Instinct Engine compilation failed"
    exit 1
fi

# Compile the demonstration program
echo "🎯 Compiling Demonstration Program..."
g++ $CXX_FLAGS melvin_instinct_engine_demo.cpp build/melvin_instinct_engine.o -o build/melvin_instinct_demo

if [ $? -eq 0 ]; then
    echo "✅ Demonstration program compiled successfully"
else
    echo "❌ Demonstration program compilation failed"
    exit 1
fi

# Run the demonstration
echo ""
echo "🚀 Running Instinct Engine Demonstration..."
echo "==========================================="
./build/melvin_instinct_demo

echo ""
echo "🎉 Build and demonstration complete!"
echo ""
echo "📁 Generated files:"
echo "- build/melvin_instinct_engine.o (object file)"
echo "- build/melvin_instinct_demo (executable)"
echo ""
echo "🔧 Integration with Melvin:"
echo "1. Include 'melvin_instinct_engine.h' in your Melvin project"
echo "2. Link with melvin_instinct_engine.o"
echo "3. Use InstinctEngine class to bias blended reasoning"
echo "4. Call get_instinct_bias() and reinforce_instinct() methods"
