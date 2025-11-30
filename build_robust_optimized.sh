#!/bin/bash

echo "🔧 Building Melvin Robust Optimized Storage + Tutor Hardening System"
echo "==================================================================="

# Compile the robust optimized system
g++ -std=c++17 -O2 -pthread -o melvin_robust_optimized \
    melvin_robust_optimized.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Robust Optimized Storage + Tutor Hardening System..."
    echo ""
    
    # Run the robust optimized system
    ./melvin_robust_optimized
else
    echo "❌ Build failed!"
    exit 1
fi
