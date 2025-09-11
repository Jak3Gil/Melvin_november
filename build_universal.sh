#!/bin/bash

echo "🔗 Building Melvin Universal Connection System..."

# Compile the universal connection system
g++ -std=c++17 -O2 -o melvin_universal melvin_universal_connections.cpp -lcurl $(pkg-config --cflags --libs jsoncpp)

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_universal"
    echo "📋 Make sure Ollama is running: ollama serve"
    echo "💡 Or test single question: ./melvin_universal 'What is a doctor?'"
    echo ""
    echo "🎯 This system applies connection-based reasoning to EVERYTHING Melvin thinks about!"
    echo "   - Semantic groups (cat → dog, bird, fish)"
    echo "   - Hierarchical relationships (cat → mammal → animal)"
    echo "   - Component relationships (notebook → note + book)"
    echo "   - Causal relationships (rain → cloud, storm)"
    echo "   - Contextual relationships (kitchen → cook, eat, food)"
    echo "   - Temporal relationships (recently learned concepts)"
    echo "   - Spatial relationships (location-based connections)"
else
    echo "❌ Build failed!"
    exit 1
fi
