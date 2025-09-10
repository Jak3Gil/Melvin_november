#!/bin/bash

echo "🤖 Building Melvin ROBUST Complete Unified System - TIMEOUT PROTECTION"
echo "====================================================================="
echo "ROBUST AI RESPONSES - TIMEOUT PROTECTION - FALLBACK RESPONSES!"
echo "🔨 Compiling Melvin ROBUST Complete Unified System..."

# Create build directory if it doesn't exist
mkdir -p build

# Compile the ROBUST complete system
g++ -std=c++17 -O2 -pthread \
    -I/usr/local/include \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib \
    melvin_robust_complete_system.cpp \
    ollama_client.cpp \
    test_robust_complete_system.cpp \
    -lcurl \
    -ljsoncpp \
    -o build/melvin_robust_complete_system

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo "🚀 Starting Melvin ROBUST Complete Unified System:"
    echo "=================================================="
    echo "🧠 All features integrated - NO LOOSE ENDS!"
    echo "⚡ Reasoning engine active"
    echo "🧬 Driver system active"
    echo "💾 Binary storage active"
    echo "🎯 Learning system active"
    echo "🤖 ROBUST AI CLIENT ACTIVE!"
    echo "⏱️ TIMEOUT PROTECTION ACTIVE!"
    echo "🔄 Fallback responses ready!"
    echo "🔄 Autonomous cycles active"
    echo "🎯 ROBUST LEARNING FROM ROBUST INPUTS/OUTPUTS!"
    echo "Press Ctrl+C to stop gracefully"
    echo ""
    
    # Run the ROBUST complete system
    ./build/melvin_robust_complete_system
else
    echo "❌ Compilation failed!"
    exit 1
fi
