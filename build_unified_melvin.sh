#!/bin/bash

# ============================================================================
# BUILD UNIFIED MELVIN BRAIN
# ============================================================================
# This script builds the unified Melvin brain with continuous thought cycle

echo "🧠 Building Unified Melvin Brain"
echo "================================"

# Create build directory
mkdir -p build

# Compile the unified Melvin brain
echo "🔨 Compiling unified Melvin brain..."

g++ -std=c++17 -O2 -Wall -Wextra \
    -I. \
    melvin_unified_brain.cpp \
    test_unified_melvin.cpp \
    -o build/test_unified_melvin \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    echo "🚀 Running Unified Melvin Brain Test:"
    echo "====================================="
    
    # Run the test
    ./build/test_unified_melvin
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Unified Melvin Brain Test Completed Successfully!"
        echo "=================================================="
        echo ""
        echo "📁 Files created:"
        echo "   - melvin_unified_brain.h (unified brain header)"
        echo "   - melvin_unified_brain.cpp (unified brain implementation)"
        echo "   - test_unified_melvin.cpp (test program)"
        echo "   - melvin_unified_test_memory/ (brain memory storage)"
        echo ""
        echo "🧠 Features implemented:"
        echo "   ✅ Continuous thought cycle"
        echo "   ✅ Binary storage with compression"
        echo "   ✅ Intelligent connection traversal"
        echo "   ✅ Dynamic node creation"
        echo "   ✅ Meta-cognitive self-evaluation"
        echo "   ✅ External interrupt handling"
        echo "   ✅ Hebbian learning"
        echo "   ✅ Memory consolidation"
        echo "   ✅ Unified interface"
        echo ""
        echo "🎯 Melvin's structure is now unified!"
    else
        echo "❌ Test execution failed!"
        exit 1
    fi
else
    echo "❌ Compilation failed!"
    exit 1
fi
