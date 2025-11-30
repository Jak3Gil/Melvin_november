#!/bin/bash

echo "🔧 Building Melvin Ultimate Unified System with Node-Travel Output"
echo "================================================================="

# Compile the ultimate unified system with output generation
g++ -std=c++17 -O2 -pthread -o melvin_ultimate_unified_with_output \
    melvin_ultimate_unified_with_output.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin Ultimate Unified with Output Generation..."
    echo ""
    echo "🎯 NEW FEATURES:"
    echo "  🧠 Node-Travel Output System"
    echo "  🔍 Reasoning → Communication Pipeline"
    echo "  📊 Response Quality Tracking"
    echo "  🔄 Tutor Feedback Integration"
    echo "  💾 Unified Memory Storage"
    echo ""
    
    # Run the ultimate unified system with output
    ./melvin_ultimate_unified_with_output
else
    echo "❌ Build failed!"
    exit 1
fi
