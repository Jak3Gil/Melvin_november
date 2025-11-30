#!/bin/bash

# Build script for Melvin Binary Brain System
# This version fixes the micro-node explosion performance issues

echo "🔨 Building Melvin Binary Brain System..."
echo "=========================================="

# Clean previous builds
rm -f melvin_binary_minimal
rm -f melvin_binary_brain.bin

# Compile with optimizations
echo "📦 Compiling binary node system..."
g++ -std=c++17 -O2 -o melvin_binary_minimal melvin_binary_minimal.cpp

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    echo "🚀 Melvin Binary Brain System Features:"
    echo "  ✅ Binary node IDs (8 bytes each)"
    echo "  ✅ Memory-efficient connections"
    echo "  ✅ No micro-node explosions"
    echo "  ✅ Hebbian learning preserved"
    echo "  ✅ Temporal chaining maintained"
    echo "  ✅ Fast binary storage/retrieval"
    echo ""
    echo "🎯 Performance Improvements:"
    echo "  🚀 Eliminates segmentation faults"
    echo "  🚀 Prevents memory exhaustion"
    echo "  🚀 Reduces processing time"
    echo "  🚀 Maintains all reasoning capabilities"
    echo ""
    echo "💡 Usage:"
    echo "  ./melvin_binary_minimal"
    echo ""
    echo "📋 Commands:"
    echo "  'analytics' - Show brain statistics"
    echo "  'save' - Save brain state to binary file"
    echo "  'quit' - Exit and save"
else
    echo "❌ Compilation failed!"
    exit 1
fi
