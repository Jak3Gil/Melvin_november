#!/bin/bash

echo "🔗 Building Melvin Intelligent Brain..."

# Compile the intelligent brain
g++ -std=c++17 -O2 -o melvin_intelligent_brain melvin_intelligent_brain.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_intelligent_brain"
    echo ""
    echo "🎯 This fixes the problems you identified:"
    echo "   ✅ Intelligent question generation (not random)"
    echo "   ✅ Semantic connection understanding (not just categories)"
    echo "   ✅ Performance optimization for large knowledge bases"
    echo "   ✅ Real curiosity-driven exploration"
    echo ""
    echo "🧠 Now Melvin will:"
    echo "   - Ask questions based on knowledge gaps"
    echo "   - Connect concepts semantically (car ↔ engine, not just car ↔ motorcycle)"
    echo "   - Understand WHY concepts connect"
    echo "   - Scale efficiently as knowledge grows"
    echo ""
    echo "🔍 Watch the difference!"
else
    echo "❌ Build failed!"
    exit 1
fi
