#!/bin/bash

# Melvin Curiosity Learning System Build Script
echo "🧠 Building Melvin Curiosity Learning System"
echo "============================================="

# Compiler flags for optimal performance
CXX_FLAGS="-std=c++17 -O3 -Wall -Wextra"

# Build Curiosity Learning System
echo "📦 Compiling Melvin Curiosity Learning..."
g++ $CXX_FLAGS -o melvin_curiosity melvin_curiosity_learning.cpp

if [ $? -eq 0 ]; then
    echo "✅ Melvin Curiosity Learning built successfully!"
    echo ""
    echo "🚀 Available system:"
    echo "   ./melvin_curiosity            # Curiosity-driven learning with binary storage"
    echo ""
    echo "📚 Usage:"
    echo "   ./melvin_curiosity \"What is a cat?\"    # Ask a question"
    echo "   ./melvin_curiosity \"What is a dog?\"    # Interactive mode"
    echo "   Type 'stats' to see learning statistics"
    echo "   Type 'quit' to exit"
    echo ""
    echo "🎯 Features:"
    echo "   ✅ Binary storage (no JSON)"
    echo "   ✅ Curiosity-tutor loop"
    echo "   ✅ Knowledge graph with connections"
    echo "   ✅ Persistent learning across sessions"
    echo "   ✅ Pure C++ implementation"
    echo ""
    echo "💾 Knowledge stored in: melvin_knowledge.bin"
else
    echo "❌ Melvin Curiosity Learning build failed!"
    exit 1
fi
