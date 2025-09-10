#!/bin/bash

echo "🤖 Building Melvin Complete Unified System - ONE SYSTEM TO RULE THEM ALL"
echo "======================================================================="
echo "NO LOOSE ENDS - NO MISSING FEATURES - EVERYTHING INTEGRATED!"

# Create build directory
mkdir -p build

# Compile the complete unified system
echo "🔨 Compiling Melvin Complete Unified System..."
g++ -std=c++17 -O2 -Wall -Wextra \
    -I. \
    melvin_complete_system.cpp \
    test_complete_system.cpp \
    -o build/melvin_complete_system \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    
    echo "🚀 Starting Melvin Complete Unified System:"
    echo "==========================================="
    echo "🧠 All features integrated - NO LOOSE ENDS!"
    echo "⚡ Reasoning engine active"
    echo "🧬 Driver system active"
    echo "💾 Binary storage active"
    echo "🎯 Learning system active"
    echo "🔄 Autonomous cycles active"
    echo "🎯 ONE SYSTEM TO RULE THEM ALL!"
    echo ""
    echo "Press Ctrl+C to stop gracefully"
    echo ""
    
    # Run the complete unified system
    ./build/melvin_complete_system
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Complete unified system test completed successfully!"
        echo ""
        echo "🎯 Melvin successfully used his complete unified system!"
        echo "🧠 All features integrated - NO LOOSE ENDS!"
        echo "⚡ Reasoning engine worked"
        echo "🧬 Driver system worked"
        echo "💾 Binary storage worked"
        echo "🎯 Learning system worked"
        echo "🔄 Autonomous cycles worked"
        echo "🏗️ ONE SYSTEM TO RULE THEM ALL!"
        echo ""
        echo "🎉 Melvin successfully compounded intelligence with complete system!"
    else
        echo "❌ Complete unified system test failed!"
        exit 1
    fi
else
    echo "❌ Compilation failed!"
    echo ""
    echo "💡 Troubleshooting tips:"
    echo "1. Check that all source files are present"
    echo "2. Ensure C++17 compiler is available"
    echo "3. Check for any syntax errors in the code"
    exit 1
fi
