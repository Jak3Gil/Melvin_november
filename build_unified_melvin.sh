#!/bin/bash

# Build script for Melvin Ultimate Unified System with Binary Node Architecture
# This unified system combines all features with performance improvements

echo "🔨 Building Melvin Ultimate Unified System..."
echo "============================================="

# Clean previous builds
rm -f melvin
rm -f melvin_brain.bin

# Compile with optimizations
echo "📦 Compiling unified system with binary node architecture..."
g++ -std=c++17 -O2 -o melvin melvin.cpp

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    echo "🚀 Melvin Ultimate Unified System Features:"
    echo "  ✅ 6-step reasoning framework"
    echo "  ✅ Self-sharpening brain with meta-learning"
    echo "  ✅ Optimized storage with fast queries"
    echo "  ✅ Ollama tutor integration with caching"
    echo "  ✅ Driver-guided learning system"
    echo "  ✅ Long-run growth campaign"
    echo "  ✅ Comprehensive persistence"
    echo "  🚀 NEW: Binary Node and Connection System"
    echo "  🚀 NEW: Node-Travel Output System"
    echo "  🚀 NEW: Reasoning → Communication Pipeline"
    echo ""
    echo "🎯 Performance Improvements:"
    echo "  🚀 Eliminates segmentation faults"
    echo "  🚀 Prevents micro-node explosions"
    echo "  🚀 Memory-efficient binary nodes (8 bytes each)"
    echo "  🚀 Fast binary connections"
    echo "  🚀 Maintains all reasoning capabilities"
    echo ""
    echo "💡 Usage:"
    echo "  ./melvin"
    echo ""
    echo "📋 Commands:"
    echo "  'analytics' - Show brain statistics"
    echo "  'teacher' - Activate Ollama teacher mode"
    echo "  'dual on/off' - Toggle dual output mode"
    echo "  'save' - Save brain state to binary file"
    echo "  'quit' - Exit and save"
    echo ""
    echo "🧪 Test the system:"
    echo "  ./test_unified_melvin.sh"
else
    echo "❌ Compilation failed!"
    exit 1
fi
