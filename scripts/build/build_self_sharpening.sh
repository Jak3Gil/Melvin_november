#!/bin/bash

echo "🔧 Building Melvin Self-Sharpening Unified Brain System"
echo "======================================================="

# Compile the self-sharpening system
g++ -std=c++17 -O2 -o melvin_self_sharpening_brain \
    melvin_self_sharpening_brain.cpp \
    -lcurl \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Self-Sharpening Unified Brain System..."
    echo ""
    
    # Run the self-sharpening system
    ./melvin_self_sharpening_brain
else
    echo "❌ Build failed!"
    exit 1
fi
