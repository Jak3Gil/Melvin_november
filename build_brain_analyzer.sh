#!/bin/bash

echo "🧠 Building Melvin Brain Analyzer..."
echo "===================================="

# Compile the brain analyzer
g++ -std=c++17 -O2 -o melvin_brain_analyzer melvin_brain_analyzer.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Brain Analyzer..."
    echo ""
    
    # Run the analyzer
    ./melvin_brain_analyzer
else
    echo "❌ Build failed!"
    exit 1
fi
