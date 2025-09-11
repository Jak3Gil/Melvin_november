#!/bin/bash

echo "🔗 Building Melvin Final Semantic Connection System..."

# Compile the final semantic connection system with working CURL
g++ -std=c++17 -O2 -o melvin_semantic_final melvin_semantic_final.cpp -lcurl $(pkg-config --cflags --libs jsoncpp)

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_semantic_final"
    echo "📋 Make sure Ollama is running: ollama serve"
    echo "💡 Or test single question: ./melvin_semantic_final 'What is a notebook?'"
else
    echo "❌ Build failed!"
    exit 1
fi
