#!/bin/bash

echo "🔧 Building Melvin Test Growth System"
echo "====================================="

# Compile the test growth system
g++ -std=c++17 -O2 -o melvin_test_growth \
    melvin_test_growth.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Test Growth System..."
    echo ""
    
    # Run the test growth system
    ./melvin_test_growth
else
    echo "❌ Build failed!"
    exit 1
fi
