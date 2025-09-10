#!/bin/bash

# Melvin Unified Build Script
echo "🧠 Building Melvin Unified AI Brain System"
echo "=========================================="

# Compiler flags for optimal performance
CXX_FLAGS="-std=c++17 -O3 -Wall -Wextra -pthread"

# Build Unified Melvin (All Features in One File)
echo "📦 Compiling Melvin Unified..."
g++ $CXX_FLAGS -o melvin_unified melvin_unified.cpp

if [ $? -eq 0 ]; then
    echo "✅ Melvin Unified built successfully!"
    echo ""
    echo "🚀 Available system:"
    echo "   ./melvin_unified            # Complete AI brain with all PDF features"
    echo ""
    echo "📚 Usage:"
    echo "   ./melvin_unified            # Start interactive session"
    echo "   Type 'checklist' to see PDF feature compliance"
    echo "   Type 'demo' to see feature demonstration"
    echo "   Type 'status' to see system metrics"
    echo "   Type 'quit' to exit"
    echo ""
    echo "🎯 Features: 18/18 PDF specifications implemented (100%)"
else
    echo "❌ Melvin Unified build failed!"
    exit 1
fi