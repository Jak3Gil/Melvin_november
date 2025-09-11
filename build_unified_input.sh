#!/bin/bash

echo "🧠 Building Melvin Unified Input Processing System..."
echo "=================================================="

# Compile the unified input processing system
g++ -std=c++17 -O2 -o melvin_unified_input_processing melvin_unified_input_processing.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Unified Input Processing System..."
    echo ""
    
    # Run the system
    ./melvin_unified_input_processing
else
    echo "❌ Build failed!"
    exit 1
fi
