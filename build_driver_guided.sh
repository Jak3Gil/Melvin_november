#!/bin/bash

echo "🔗 Building Melvin Driver-Guided Brain..."

# Compile the driver-guided brain
g++ -std=c++17 -O2 -o melvin_driver_guided melvin_driver_guided.cpp

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_driver_guided"
    echo ""
    echo "🎯 This addresses your key insights:"
    echo "   ✅ Melvin's circular questioning is GOOD - he explores deeply"
    echo "   ✅ Drivers give DIRECTION to exploration, not prevent it"
    echo "   ✅ Knowledge is PERSISTED to files (not lost on restart)"
    echo "   ✅ Drivers guide QUESTION TYPES, not question frequency"
    echo ""
    echo "🧠 Now Melvin will:"
    echo "   - Ask the SAME concept many times (deep exploration)"
    echo "   - But vary the TYPE of questions based on drivers"
    echo "   - Save all learning to melvin_knowledge.json"
    echo "   - Evolve his drivers based on experience"
    echo ""
    echo "🎭 Driver Types:"
    echo "   - Survival: Safety, dangers, protection questions"
    echo "   - Curiosity: What, how, why, when questions"
    echo "   - Efficiency: Optimization, improvement questions"
    echo "   - Social: People, connection, help questions"
    echo "   - Consistency: Relationship, contradiction questions"
    echo ""
    echo "🔍 Watch how drivers guide his exploration direction!"
else
    echo "❌ Build failed!"
    exit 1
fi
