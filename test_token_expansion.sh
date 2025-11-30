#!/bin/bash

echo "🔬 Testing Melvin's Token-to-Concept Expansion System"
echo "===================================================="
echo ""

echo "🎯 Testing micro-node expansion commands..."
echo "micro nodes on" | ./melvin
echo ""

echo "🧠 Testing input expansion with a complex sentence..."
echo "Integrated Information Theory says consciousness arises from information integration." | ./melvin
echo ""

echo "🎓 Testing Ollama response expansion..."
echo "teacher" | ./melvin
echo "What is consciousness?" | ./melvin
echo "quit" | ./melvin
echo ""

echo "🔍 Testing traversal through micro-nodes..."
echo "What is information?" | ./melvin
echo ""

echo "🎯 Disabling micro-node expansion..."
echo "micro nodes off" | ./melvin
echo ""

echo "✅ Token expansion system test complete!"
echo ""
echo "Expected behavior:"
echo "- Input sentences should be broken into micro-nodes (words, pairs, triples)"
echo "- Ollama responses should be exploded into many granular nodes"
echo "- Traversal should include both macro concepts and micro-nodes"
echo "- Dense linking should connect related micro-nodes"
