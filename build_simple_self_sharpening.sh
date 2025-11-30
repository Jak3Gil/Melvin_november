#!/bin/bash

echo "🔧 Building Melvin Simple Self-Sharpening Brain System"
echo "======================================================"

# Compile the simple self-sharpening system
g++ -std=c++17 -O2 -o melvin_simple_self_sharpening \
    melvin_simple_self_sharpening.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Simple Self-Sharpening Brain System..."
    echo ""
    
    # Run the simple self-sharpening system
    ./melvin_simple_self_sharpening
else
    echo "❌ Build failed!"
    exit 1
fi
