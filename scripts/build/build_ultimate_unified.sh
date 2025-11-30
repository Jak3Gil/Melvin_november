#!/bin/bash

echo "🔧 Building Melvin Ultimate Unified System"
echo "=========================================="

# Compile the ultimate unified system
g++ -std=c++17 -O2 -pthread -o melvin_ultimate_unified \
    melvin_ultimate_unified.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Ultimate Unified System..."
    echo ""
    
    # Run the ultimate unified system
    ./melvin_ultimate_unified
else
    echo "❌ Build failed!"
    exit 1
fi
