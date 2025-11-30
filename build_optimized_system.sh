#!/bin/bash

echo "🔧 Building Melvin Optimized Storage + Tutor Hardening System"
echo "============================================================="

# Compile the optimized system
g++ -std=c++17 -O2 -pthread -o melvin_optimized_storage_tutor \
    melvin_optimized_storage_tutor.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Optimized Storage + Tutor Hardening System..."
    echo ""
    
    # Run the optimized system
    ./melvin_optimized_storage_tutor
else
    echo "❌ Build failed!"
    exit 1
fi
