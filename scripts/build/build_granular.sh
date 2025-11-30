#!/bin/bash

echo "🔗 Building Melvin Granular Node System..."

# Compile the granular node system
g++ -std=c++17 -O2 -o melvin_granular_nodes melvin_granular_nodes.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_granular_nodes"
    echo ""
    echo "🎯 This demonstrates granular node breakdown:"
    echo "   - Breaks 'cat' into: cat + small + domesticated + carnivorous + mammal + soft fur + short snout + retractable claws"
    echo "   - Each component becomes a separate, reusable node"
    echo "   - 'mammal' can be reused for dog, lion, tiger, etc."
    echo "   - 'carnivorous' can be reused across all meat-eaters"
    echo "   - 'domesticated' can be reused for all pets"
    echo ""
    echo "🧠 Benefits:"
    echo "   - Reusable components instead of monolithic definitions"
    echo "   - Richer connections through shared nodes"
    echo "   - More efficient learning (reuse vs. recreate)"
    echo "   - Better generalization across concepts"
else
    echo "❌ Build failed!"
    exit 1
fi
