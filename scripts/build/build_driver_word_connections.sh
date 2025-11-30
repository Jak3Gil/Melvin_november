#!/bin/bash

echo "🔗 Building Melvin Driver-Guided Word Connection Brain..."

# Compile the driver-guided word connection brain
g++ -std=c++17 -O2 -o melvin_driver_word_connections melvin_driver_word_connections.cpp -lcurl

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_driver_word_connections"
    echo ""
    echo "🎯 This combines the best of both systems:"
    echo "   ✅ Driver-guided questioning (varied question types)"
    echo "   ✅ Real Ollama answers (not simulated)"
    echo "   ✅ Word connection analysis (connect words that appear together)"
    echo "   ✅ Knowledge persistence (save everything with driver info)"
    echo ""
    echo "🧠 How it works:"
    echo "   1. Melvin's drivers determine question type (survival, curiosity, etc.)"
    echo "   2. Melvin asks driver-guided questions to Ollama"
    echo "   3. Ollama gives real answers"
    echo "   4. Melvin connects words that appear together in answers"
    echo "   5. Drivers evolve based on experience"
    echo "   6. Everything saved with driver context"
    echo ""
    echo "🎭 Driver Types:"
    echo "   - Survival: Safety, dangers, protection questions"
    echo "   - Curiosity: What, how, why, when questions"
    echo "   - Efficiency: Optimization, improvement questions"
    echo "   - Social: People, connection, help questions"
    echo "   - Consistency: Relationship, contradiction questions"
    echo ""
    echo "🔍 Watch how drivers guide questions AND word connections!"
else
    echo "❌ Build failed!"
    exit 1
fi
