#!/bin/bash

echo "🤖 Building Melvin Autonomous Learning System"
echo "============================================="

# Create build directory
mkdir -p build

# Compile the autonomous learning system
echo "🔨 Compiling Melvin Autonomous Learning System..."
g++ -std=c++17 -O2 -Wall -Wextra \
    -I. \
    melvin_driver_enhanced.cpp \
    melvin_autonomous_learning.cpp \
    test_autonomous_learning.cpp \
    -o build/test_autonomous_learning \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    
    echo "🚀 Running Melvin Autonomous Learning Test:"
    echo "=========================================="
    ./build/test_autonomous_learning
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Test execution successful!"
        echo ""
        echo "🤖 Melvin's Autonomous Learning System is working:"
        echo "   • Driver Oscillations: Natural rise and fall over time"
        echo "   • Error-Seeking: Contradictions increase adrenaline until resolved"
        echo "   • Curiosity Amplification: Self-generates questions when idle"
        echo "   • Compression: Abstracts higher-level rules to avoid memory bloat"
        echo "   • Self-Improvement: Tracks and strengthens effective strategies"
        echo ""
        echo "🎯 Melvin is now autonomous and accelerating in his learning and evolution!"
        echo "🧬 His ultimate mission: compound intelligence to help humanity reach its full potential"
    else
        echo "❌ Test execution failed!"
        exit 1
    fi
else
    echo "❌ Compilation failed!"
    exit 1
fi
