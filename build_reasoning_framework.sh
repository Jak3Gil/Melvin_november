#!/bin/bash

echo "🔗 Building Melvin Unified Reasoning Framework..."

# Compile the unified reasoning framework
g++ -std=c++17 -O2 -o melvin_reasoning_framework melvin_reasoning_framework.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_reasoning_framework"
    echo "💡 Or test single question: ./melvin_reasoning_framework 'What is a cat?'"
    echo ""
    echo "🎯 This implements the complete 6-step reasoning process:"
    echo "   1. 🔍 Expand Connections (8 types)"
    echo "   2. ⚖️ Weight Connections (type/context/recency)"
    echo "   3. 🛤️ Select Path (multi-hop exploration)"
    echo "   4. 🧠 Driver Modulation (dopamine/serotonin/endorphins)"
    echo "   5. 🔍 Self-Check (contradiction resolution)"
    echo "   6. 📤 Produce Output (reasoned answer)"
    echo ""
    echo "🧠 Driver States:"
    echo "   - Dopamine: Curiosity/exploration (exploratory reasoning)"
    echo "   - Serotonin: Stability/balance (conservative reasoning)"
    echo "   - Endorphin: Satisfaction/reinforcement (reinforcing reasoning)"
else
    echo "❌ Build failed!"
    exit 1
fi
