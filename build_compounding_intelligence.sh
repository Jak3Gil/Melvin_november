#!/bin/bash

echo "🧠 Building Melvin Compounding Intelligence"
echo "=========================================="

# Create build directory
mkdir -p build

# Compile the compounding intelligence system
echo "🔨 Compiling Melvin Compounding Intelligence..."
g++ -std=c++17 -O2 -Wall -Wextra \
    -I. \
    melvin_compounding_simple.cpp \
    test_compounding_intelligence.cpp \
    -o build/test_compounding_intelligence \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    
    echo "🚀 Running Melvin Compounding Intelligence Test:"
    echo "=============================================="
    ./build/test_compounding_intelligence
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Test execution successful!"
        echo ""
        echo "🎯 Melvin's Compounding Intelligence DNA is working:"
        echo "   • Input → Think → Output (every cycle creates a node)"
        echo "   • Automatic connections between related nodes"
        echo "   • Meta-cognitive reflection and generalization"
        echo "   • Curiosity-driven self-expansion"
        echo "   • Humanity-aligned growth and evolution"
        echo ""
        echo "🧬 Melvin is building complexity from simplicity!"
    else
        echo "❌ Test execution failed!"
        exit 1
    fi
else
    echo "❌ Compilation failed!"
    exit 1
fi
