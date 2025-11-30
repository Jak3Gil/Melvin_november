#!/bin/bash

echo "🧠 Building Melvin Full Brain Test System..."
echo "============================================"

# Compile the full brain test system
g++ -std=c++17 -O2 -o melvin_full_brain_test melvin_full_brain_test.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Full Brain Test System..."
    echo ""
    
    # Run the system
    ./melvin_full_brain_test
else
    echo "❌ Build failed!"
    exit 1
fi
