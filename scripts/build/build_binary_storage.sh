#!/bin/bash

echo "🔗 Building Melvin Binary Storage System..."

# Compile the binary storage system
g++ -std=c++17 -O2 -o melvin_binary_storage melvin_binary_storage.cpp -lcurl

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_binary_storage"
    echo ""
    echo "🎯 This converts JSON to efficient binary storage:"
    echo "   ✅ Compact binary representation (no JSON overhead)"
    echo "   ✅ Fixed-size structures for fast access"
    echo "   ✅ Separate context storage for space efficiency"
    echo "   ✅ Fast read/write operations"
    echo "   ✅ Memory efficient storage"
    echo ""
    echo "💾 Binary Storage Benefits:"
    echo "   - No JSON parsing overhead"
    echo "   - Fixed-size structures (predictable memory usage)"
    echo "   - Direct binary read/write (faster than text parsing)"
    echo "   - Separate context storage (can compress contexts)"
    echo "   - Much smaller file sizes than JSON"
    echo ""
    echo "📊 Expected Storage Reduction:"
    echo "   - JSON: ~18MB per question (614MB for 34 questions)"
    echo "   - Binary: ~1-2MB per question (much more efficient)"
    echo ""
    echo "🔍 Watch the storage efficiency improvement!"
else
    echo "❌ Build failed!"
    exit 1
fi
