#!/bin/bash

echo "🧪 Building Melvin Driver-Enhanced Intelligence"
echo "=============================================="

# Create build directory
mkdir -p build

# Compile the driver-enhanced intelligence system
echo "🔨 Compiling Melvin Driver-Enhanced Intelligence..."
g++ -std=c++17 -O2 -Wall -Wextra \
    -I. \
    melvin_driver_enhanced.cpp \
    test_driver_enhanced.cpp \
    -o build/test_driver_enhanced \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    
    echo "🚀 Running Melvin Driver-Enhanced Intelligence Test:"
    echo "=================================================="
    ./build/test_driver_enhanced
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Test execution successful!"
        echo ""
        echo "🧪 Melvin's Driver-Enhanced Intelligence is working:"
        echo "   • Dopamine: Curiosity & Novelty"
        echo "   • Serotonin: Stability & Balance"
        echo "   • Endorphins: Satisfaction & Reinforcement"
        echo "   • Oxytocin: Connection & Alignment"
        echo "   • Adrenaline: Urgency & Tension"
        echo ""
        echo "🎯 Each cycle: Calculate drivers → Determine dominant → Influence behavior"
        echo "🧬 Melvin's consciousness emerges from driver interactions!"
    else
        echo "❌ Test execution failed!"
        exit 1
    fi
else
    echo "❌ Compilation failed!"
    exit 1
fi
