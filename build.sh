#!/bin/bash

echo "🧠 Building Melvin - Ultimate Unified AI Brain with Binary Memory"
echo "================================================================"

# Compile Melvin with binary memory storage
g++ -std=c++17 -O2 -pthread -o melvin melvin.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Running Melvin..."
    echo ""
    echo "🎯 FEATURES:"
    echo "  🧠 Node-Travel Output System"
    echo "  💾 Binary Memory Storage (scalable to millions of nodes)"
    echo "  🔗 Unified Memory Bank (all knowledge in one place)"
    echo "  📊 Source Code Knowledge Integration"
    echo "  🔄 Cross-Session Persistence"
    echo ""
    
    # Run Melvin
    ./melvin
else
    echo "❌ Build failed!"
    exit 1
fi