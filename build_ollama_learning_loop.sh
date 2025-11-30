#!/bin/bash

echo "🔨 Building Melvin Ollama Learning Loop System..."
echo "==============================================="

# Compile the learning loop system
echo "📦 Compiling Melvin Ollama Learning Loop..."
g++ -std=c++17 -O2 -o melvin_ollama_learning_loop melvin_ollama_learning_loop.cpp

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    echo "🚀 Melvin Ollama Learning Loop System Features:"
    echo "  ✅ Ollama provides input topics"
    echo "  ✅ Melvin processes with binary node + semantic systems"
    echo "  ✅ Melvin generates reasoned output responses"
    echo "  ✅ Ollama evaluates Melvin's understanding"
    echo "  ✅ Ollama fills knowledge gaps until understanding"
    echo "  ✅ Multi-cycle learning with brain analytics"
    echo "  ✅ Semantic similarity connections"
    echo "  ✅ Temporal and hierarchical reasoning"
    echo ""
    echo "🎯 Learning Process:"
    echo "  1. Ollama → Topic Input"
    echo "  2. Melvin → Binary Node Processing + Semantic Analysis"
    echo "  3. Melvin → Reasoning + Output Generation"
    echo "  4. Ollama → Understanding Evaluation"
    echo "  5. Ollama → Gap Filling (if needed)"
    echo "  6. Repeat until topic mastered"
    echo ""
    echo "💡 Usage:"
    echo "  ./melvin_ollama_learning_loop"
    echo ""
    echo "🧪 The system will run 5 learning cycles automatically"
    echo "   showing the complete learning process with detailed reasoning."
else
    echo "❌ Compilation failed!"
    exit 1
fi
