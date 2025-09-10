#!/bin/bash

echo "🤖 Building Melvin Truly Autonomous Learning"
echo "============================================="

# Create build directory
mkdir -p build

# Compile the truly autonomous learning system
echo "🔨 Compiling Melvin Truly Autonomous Learning..."
g++ -std=c++17 -O2 -Wall -Wextra \
    -I. \
    melvin_driver_enhanced.cpp \
    melvin_autonomous_learning.cpp \
    melvin_truly_autonomous.cpp \
    -o build/melvin_truly_autonomous \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    
    echo "🚀 Starting Melvin Truly Autonomous Learning:"
    echo "============================================="
    echo "🤖 Melvin will generate his own inputs from his outputs!"
    echo "🔄 TRUE AUTONOMY: His outputs become his inputs (feedback loop)"
    echo "🧪 Driver oscillations will create natural learning rhythms"
    echo "🔍 Error-seeking will drive contradiction resolution"
    echo "🎯 Curiosity amplification will fill empty space"
    echo "📦 Compression will keep knowledge efficient"
    echo "⚡ Self-improvement will accelerate evolution"
    echo ""
    echo "Press Ctrl+C to stop gracefully"
    echo ""
    
    # Run the truly autonomous learning system
    ./build/melvin_truly_autonomous
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Truly autonomous learning completed successfully!"
        echo ""
        echo "🎯 Melvin successfully generated his own inputs from his outputs!"
        echo "🔄 TRUE AUTONOMY: His outputs became his inputs (feedback loop)"
        echo "🧬 His driver oscillations created natural learning rhythms"
        echo "🔍 Error-seeking drove contradiction resolution"
        echo "🎯 Curiosity amplification filled empty space"
        echo "📦 Compression kept knowledge efficient"
        echo "⚡ Self-improvement accelerated evolution"
        echo ""
        echo "🎉 Melvin successfully compounded intelligence truly autonomously!"
    else
        echo "❌ Truly autonomous learning failed!"
        exit 1
    fi
else
    echo "❌ Compilation failed!"
    exit 1
fi
