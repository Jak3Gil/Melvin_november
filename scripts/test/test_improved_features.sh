#!/bin/bash

echo "🧠 Testing Melvin's Improved Answer Generation and Evaluation"
echo "============================================================"
echo ""

echo "🎯 Testing sentence composer..."
echo "What is consciousness?" | ./melvin
echo ""

echo "🧠 Testing multi-pass thinking..."
echo "dual thinking on" | ./melvin
echo "How does photosynthesis work?" | ./melvin
echo ""

echo "🎯 Testing evaluation with Effort criterion..."
echo "evaluation on" | ./melvin
echo "evaluate me" | ./melvin
echo ""

echo "📊 Testing confidence-based responses..."
echo "What is creativity?" | ./melvin
echo ""

echo "✅ Improved features test complete!"
echo ""
echo "Expected improvements:"
echo "- No more shell escaping errors in evaluation prompts"
echo "- Full sentences instead of fragments (e.g., 'consciousness can be understood as awareness, self, and environment')"
echo "- Multi-pass thinking shows fast vs deep reasoning paths"
echo "- Evaluation includes Effort criterion (4 criteria total)"
echo "- Fallback scores when Ollama evaluation fails"
echo "- Confidence-based response qualifiers"
