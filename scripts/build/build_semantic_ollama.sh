#!/bin/bash

echo "🔗 Building Melvin Semantic Connection System with Real Ollama..."

# Compile the semantic connection system with real Ollama
g++ -std=c++17 -O2 -o melvin_semantic_ollama melvin_semantic_ollama.cpp -lcurl $(pkg-config --cflags --libs jsoncpp)

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_semantic_ollama"
    echo "📋 Make sure Ollama is running: ollama serve"
else
    echo "❌ Build failed!"
    exit 1
fi
