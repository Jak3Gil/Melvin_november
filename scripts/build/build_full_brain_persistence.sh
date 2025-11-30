#!/bin/bash

echo "🧠 Building Melvin Full Brain Persistence + Ollama Tutor..."
echo "========================================================="

# Check if libcurl is available
if ! pkg-config --exists libcurl; then
    echo "❌ libcurl not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install curl
    else
        echo "❌ Homebrew not found. Please install libcurl manually."
        exit 1
    fi
fi

# Get libcurl flags
CURL_FLAGS=$(pkg-config --cflags --libs libcurl)

# Compile the full brain persistence system
g++ -std=c++17 -O2 $CURL_FLAGS -o melvin_full_brain_persistence_ollama melvin_full_brain_persistence_ollama.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Full Brain Persistence + Ollama Tutor..."
    echo ""
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is running and accessible"
    else
        echo "⚠️  Ollama not running - tutor features will be simulated"
    fi
    
    # Run the system
    ./melvin_full_brain_persistence_ollama
else
    echo "❌ Build failed!"
    exit 1
fi
