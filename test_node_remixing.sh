#!/bin/bash

echo "🧠 Testing Melvin's Node Remixing and Context-Based Sentence Synthesis"
echo "====================================================================="
echo ""

echo "🎯 Testing sentence storage and remixing..."
echo "teacher" | ./melvin
echo "What is consciousness?" | ./melvin
echo ""

echo "🧠 Testing multi-node traversal with remixed sentences..."
echo "dual thinking on" | ./melvin
echo "How does photosynthesis work?" | ./melvin
echo ""

echo "📝 Testing paragraph-level explanations..."
echo "What is creativity?" | ./melvin
echo ""

echo "🔄 Testing variety in remixed responses..."
echo "What is consciousness?" | ./melvin
echo ""

echo "✅ Node remixing test complete!"
echo ""
echo "Expected improvements:"
echo "- Sentences are stored in node.seenSentences"
echo "- Remixed sentences combine clauses from multiple sources"
echo "- Paragraph-level explanations with transition phrases"
echo "- Variety in responses through clause shuffling"
echo "- Confidence-based sentence selection"
echo "- Original 'Frankensteined' knowledge paths"
