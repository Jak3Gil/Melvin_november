#!/bin/bash

echo "🧠 Building Melvin Ultimate - The Definitive AI Brain System"
echo "============================================================"

# Compile Melvin Ultimate with all integrated features
g++ -std=c++17 -O2 -pthread -o melvin_ultimate melvin_ultimate.cpp

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
    echo "  🔮 Ollama Tutor Integration (external oracle support)"
    echo "  💾 Unified Memory Bank (all knowledge in one place)"
    echo ""
    echo "💡 Commands:"
    echo "  - Ask questions normally"
    echo "  - Type 'explore' for autonomous learning"
    echo "  - Type 'analytics' to see brain state"
    echo "  - Type 'quit' to exit"
    echo ""
    
    # Run Melvin Ultimate
    ./melvin_ultimate
else
    echo "❌ Build failed!"
    exit 1
fi
