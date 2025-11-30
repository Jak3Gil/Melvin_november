#!/bin/bash

echo "🔗 Building Melvin Semantic Connection System..."

# Compile the semantic connection demo
g++ -std=c++17 -O2 -o melvin_semantic_demo melvin_semantic_connections.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_semantic_demo"
else
    echo "❌ Build failed!"
    exit 1
fi
