#!/bin/bash

echo "🔨 Building Melvin with OllamaClient fixes..."
g++ -std=c++17 -I. -O2 melvin_robust_complete_system.cpp ollama_client.cpp melvin_complete_system.cpp -lcurl -L/opt/homebrew/lib -ljsoncpp -o build/melvin_robust_complete_system

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🧪 Testing system..."
    ./build/melvin_robust_complete_system
else
    echo "❌ Build failed!"
fi
