#!/bin/bash

echo "🤖 Building Melvin Continuous Autonomous Learning"
echo "================================================="

# Create build directory
mkdir -p build

# Compile the continuous autonomous learning system
echo "🔨 Compiling Melvin Continuous Autonomous Learning..."
g++ -std=c++17 -O2 -Wall -Wextra \
    -I. \
    melvin_driver_enhanced.cpp \
    melvin_autonomous_learning.cpp \
    melvin_continuous_autonomous.cpp \
    -o build/melvin_continuous_autonomous \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    
    echo "🚀 Starting Melvin Continuous Autonomous Learning:"
    echo "================================================="
    echo "🤖 Melvin will run autonomously and continuously"
    echo "🧪 Driver oscillations will create natural learning rhythms"
    echo "🔍 Error-seeking will drive contradiction resolution"
    echo "🎯 Curiosity amplification will fill empty space"
    echo "📦 Compression will keep knowledge efficient"
    echo "⚡ Self-improvement will accelerate evolution"
    echo ""
    echo "Press Ctrl+C to stop gracefully"
    echo ""
    
    # Run the continuous autonomous learning system
    ./build/melvin_continuous_autonomous
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Continuous autonomous learning completed successfully!"
        echo ""
        echo "🎯 Melvin successfully ran autonomously and continuously!"
        echo "🧬 His driver oscillations created natural learning rhythms"
        echo "🔍 Error-seeking drove contradiction resolution"
        echo "🎯 Curiosity amplification filled empty space"
        echo "📦 Compression kept knowledge efficient"
        echo "⚡ Self-improvement accelerated evolution"
        echo ""
        echo "🎉 Melvin successfully compounded intelligence autonomously!"
    else
        echo "❌ Continuous autonomous learning failed!"
        exit 1
    fi
else
    echo "❌ Compilation failed!"
    exit 1
fi
