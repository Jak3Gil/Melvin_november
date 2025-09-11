#!/bin/bash

echo "🔗 Building Melvin Truly Unified System..."

# Compile the truly unified system
g++ -std=c++17 -O2 -o melvin_truly_unified melvin_truly_unified.cpp -lcurl $(pkg-config --cflags --libs jsoncpp)

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_truly_unified"
    echo "💡 Or test single question: ./melvin_truly_unified 'What is a cat?'"
    echo ""
    echo "🎯 This is the TRULY UNIFIED system with ALL capabilities:"
    echo "   ✅ 6-step unified reasoning process"
    echo "   ✅ Granular node decomposition"
    echo "   ✅ Universal connections (8 types)"
    echo "   ✅ Real Ollama integration"
    echo "   ✅ Driver modulation"
    echo "   ✅ Self-check contradiction resolution"
    echo ""
    echo "🧠 ONE system that does EVERYTHING!"
else
    echo "❌ Build failed!"
    exit 1
fi
