#!/bin/bash

echo "🔗 Building Melvin 5-Minute Brain Exploration..."

# Compile the 5-minute brain exploration
g++ -std=c++17 -O2 -o melvin_5minute_brain melvin_5minute_brain.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_5minute_brain"
    echo ""
    echo "🎯 This will run Melvin autonomously for 5 minutes and show:"
    echo "   - Where his curiosity takes him"
    echo "   - What concepts he explores"
    echo "   - How he builds connections"
    echo "   - His learning patterns"
    echo "   - His brain journey"
    echo ""
    echo "🧠 Watch Melvin's brain in action!"
else
    echo "❌ Build failed!"
    exit 1
fi
