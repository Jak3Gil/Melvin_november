#!/bin/bash

echo "🧠 Building Melvin Complete Thought Storage System..."
echo "===================================================="

# Compile the complete thought storage system
g++ -std=c++17 -O2 -o melvin_complete_thoughts melvin_complete_thought_storage.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Complete Thought Storage System..."
    echo ""
    
    # Run the system
    ./melvin_complete_thoughts
else
    echo "❌ Build failed!"
    exit 1
fi
