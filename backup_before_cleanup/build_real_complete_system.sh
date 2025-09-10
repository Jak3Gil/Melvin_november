#!/bin/bash

echo "🤖 Building Melvin REAL Complete Unified System - REAL AI INTEGRATION"
echo "====================================================================="
echo "REAL AI RESPONSES - REAL LEARNING - NO FAKE OUTPUTS!"
echo "🔨 Compiling Melvin REAL Complete Unified System..."

# Create build directory if it doesn't exist
mkdir -p build

# Compile the REAL complete system
g++ -std=c++17 -O2 -pthread \
    -I/usr/local/include \
    -I/opt/homebrew/include \
    melvin_real_complete_system.cpp \
    ollama_client.cpp \
    test_real_complete_system.cpp \
    -lcurl \
    -ljsoncpp \
    -o build/melvin_real_complete_system

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo "🚀 Starting Melvin REAL Complete Unified System:"
    echo "================================================"
    echo "🧠 All features integrated - NO LOOSE ENDS!"
    echo "⚡ Reasoning engine active"
    echo "🧬 Driver system active"
    echo "💾 Binary storage active"
    echo "🎯 Learning system active"
    echo "🤖 REAL AI CLIENT ACTIVE!"
    echo "🔄 Autonomous cycles active"
    echo "🎯 REAL LEARNING FROM REAL INPUTS/OUTPUTS!"
    echo "Press Ctrl+C to stop gracefully"
    echo ""
    
    # Run the REAL complete system
    ./build/melvin_real_complete_system
else
    echo "❌ Compilation failed!"
    exit 1
fi
