#!/bin/bash

echo "🔧 Building Melvin Truly Unified Reasoning System"
echo "================================================"

# Compile the truly unified system
g++ -std=c++17 -O2 -o melvin_truly_unified_reasoning \
    melvin_truly_unified_reasoning.cpp \
    -lcurl \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Truly Unified Reasoning System..."
    echo ""
    
    # Run the truly unified system
    ./melvin_truly_unified_reasoning
else
    echo "❌ Build failed!"
    exit 1
fi