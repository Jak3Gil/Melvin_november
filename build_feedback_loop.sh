#!/bin/bash

echo "🔗 Building Melvin Feedback Loop Engine..."

# Compile the feedback loop engine
g++ -std=c++17 -O2 -o melvin_feedback_loop melvin_feedback_loop.cpp -lcurl

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Run with: ./melvin_feedback_loop"
    echo ""
    echo "🎯 This implements the user's feedback loop prompt:"
    echo "   ✅ Driver-guided questioning (curiosity, stability, reinforcement)"
    echo "   ✅ Connection-driven refinement (use existing connections to generate new questions)"
    echo "   ✅ Autonomous feedback loops (answers → connections → new questions)"
    echo "   ✅ Dynamic driver evolution (drivers adapt based on experience)"
    echo ""
    echo "🧠 How the feedback loop works:"
    echo "   1. Generate questions based on driver state + existing connections"
    echo "   2. Ask questions to Ollama and collect answers"
    echo "   3. Extract words and build connections with strength scores"
    echo "   4. Evolve drivers based on experience (novelty, contradictions, confirmations)"
    echo "   5. Expand concept vocabulary from new answers"
    echo "   6. Repeat cycle automatically"
    echo ""
    echo "🎭 Driver Types:"
    echo "   - Curiosity: Exploratory, open-ended 'what/why/how' questions"
    echo "   - Stability: Clarifying, contradiction-checking questions"
    echo "   - Reinforcement: Summarizing, practical 'how to use' questions"
    echo ""
    echo "🔄 Connection-Driven Refinement:"
    echo "   - Reviews strong word connections (high co-occurrence)"
    echo "   - Turns connections into new questions"
    echo "   - Detects contradictions and probes them"
    echo ""
    echo "🔍 Watch Melvin evolve autonomously through feedback loops!"
else
    echo "❌ Build failed!"
    exit 1
fi
