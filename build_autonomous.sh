#!/bin/bash

echo "🔗 Building Melvin Autonomous Granular System..."

# Compile the autonomous granular system
g++ -std=c++17 -O2 -o melvin_autonomous_granular melvin_autonomous_granular.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_autonomous_granular"
    echo ""
    echo "🎯 This proves Melvin can decompose ANY topic autonomously:"
    echo "   - Animals: elephant, butterfly"
    echo "   - Technology: computer, smartphone"  
    echo "   - Food: pizza, salad"
    echo "   - Transportation: airplane, bicycle"
    echo ""
    echo "🧠 Melvin's Autonomous Capabilities:"
    echo "   - Extracts words from any definition"
    echo "   - Categorizes words by learned patterns"
    echo "   - Creates reusable component nodes"
    echo "   - Discovers relationships between components"
    echo "   - Tracks reuse across different topics"
    echo ""
    echo "🔍 NO HELP FROM ME - Melvin does everything on his own!"
else
    echo "❌ Build failed!"
    exit 1
fi
