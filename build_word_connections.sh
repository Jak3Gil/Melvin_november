#!/bin/bash

echo "🔗 Building Melvin Word Connection Brain..."

# Compile the word connection brain
g++ -std=c++17 -O2 -o melvin_word_connections melvin_word_connections.cpp -lcurl

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_word_connections"
    echo ""
    echo "🎯 This does exactly what you want:"
    echo "   ✅ Melvin asks Ollama questions"
    echo "   ✅ Ollama gives real answers"
    echo "   ✅ Melvin connects words that appear together"
    echo "   ✅ Simple word-to-word connections"
    echo "   ✅ Saves everything to melvin_word_connections.json"
    echo ""
    echo "🧠 How it works:"
    echo "   1. Melvin asks: 'What is a car?'"
    echo "   2. Ollama answers: 'A car is a vehicle with wheels and engine'"
    echo "   3. Melvin connects: car ↔ vehicle, car ↔ wheels, car ↔ engine, etc."
    echo "   4. Saves all connections with counts"
    echo ""
    echo "🔍 Watch the real word connections!"
else
    echo "❌ Build failed!"
    exit 1
fi
