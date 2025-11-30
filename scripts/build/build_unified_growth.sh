#!/bin/bash

echo "🔧 Building Melvin Unified Growth, Explainability, and Reliability System"
echo "========================================================================="

# Compile the unified growth system
g++ -std=c++17 -O2 -o melvin_unified_growth_system \
    melvin_unified_growth_system.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Unified Growth System..."
    echo ""
    
    # Run the unified growth system
    ./melvin_unified_growth_system
else
    echo "❌ Build failed!"
    exit 1
fi
