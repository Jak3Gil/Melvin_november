#!/bin/bash

echo "🎯 Testing Melvin's Dynamic Review Evaluation System"
echo "==================================================="
echo ""

echo "🎯 Testing evaluation commands..."
echo "evaluation on" | ./melvin
echo ""

echo "🧠 Testing dynamic evaluation with a concept..."
echo "evaluate me" | ./melvin
echo ""

echo "🎓 Testing evaluation with teacher mode..."
echo "teacher" | ./melvin
echo "What is consciousness?" | ./melvin
echo "quit" | ./melvin
echo ""

echo "🎯 Disabling evaluation mode..."
echo "evaluation off" | ./melvin
echo ""

echo "✅ Dynamic evaluation system test complete!"
echo ""
echo "Expected behavior:"
echo "- Ollama will ask Melvin the same question in different ways"
echo "- Ollama will score Melvin's responses on accuracy, completeness, and coherence"
echo "- Melvin's confidence scores will be updated based on evaluation results"
echo "- Review cycles will include dynamic evaluation for real assessment"
