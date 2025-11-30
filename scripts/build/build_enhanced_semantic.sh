#!/bin/bash

echo "🔗 Building Melvin Enhanced Semantic Connection System..."

# Compile the enhanced semantic connection system
g++ -std=c++17 -O2 -o melvin_enhanced_semantic melvin_enhanced_semantic.cpp ollama_client.cpp -lcurl $(pkg-config --cflags --libs jsoncpp)

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_enhanced_semantic"
    echo "📋 Make sure Ollama is running: ollama serve"
    echo "💡 Or test single question: ./melvin_enhanced_semantic 'What is a notebook?'"
else
    echo "❌ Build failed!"
    exit 1
fi
