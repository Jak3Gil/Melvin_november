#!/bin/bash

echo "🧠 Building Melvin - The Ultimate AI Brain System"
echo "================================================="

# Compile Melvin with all integrated features
g++ -std=c++17 -O2 -pthread -o melvin melvin.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Ultimate..."
    echo ""
    echo "🎯 INTEGRATED FEATURES:"
    echo "  🧠 Binary Memory Storage (scalable to millions)"
    echo "  🔍 Node-Travel Output System (reasoning → communication)"
    echo "  🎭 5-Neurotransmitter Driver System (personality + adaptive)"
    echo "  ❓ Curiosity Loop (auto-ask Ollama when uncertain)"
    echo "  🤖 Autonomous Exploration (self-directed learning)"
    echo "  🧩 Semantic Analysis (word decomposition + relationships)"
    echo "  📊 Brain State Analytics (introspection + visualization)"
    echo "  📚 Source Code Knowledge Integration (compile-time concepts)"
    echo "  🔗 Unified Memory Bank (all knowledge in one place)"
    echo ""
    echo "💡 COMMANDS:"
    echo "  - Ask any question"
    echo "  - Type 'analytics' to see brain state"
    echo "  - Type 'autonomous' for self-exploration"
    echo "  - Type 'save' to save brain state"
    echo "  - Type 'load' to load brain state"
    echo "  - Type 'quit' to exit"
    echo ""
    ./melvin
else
    echo "❌ Build failed!"
    exit 1
fi
