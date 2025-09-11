#!/bin/bash

echo "🔧 Building Melvin Unified Complete System"
echo "=========================================="

# Compile the unified system
g++ -std=c++17 -O2 -o melvin_unified_complete \
    melvin_unified_complete_system.cpp \
    -lcurl \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Unified Complete System..."
    echo ""
    
    # Run the unified system
    ./melvin_unified_complete
else
    echo "❌ Build failed!"
    exit 1
fi
