#!/bin/bash

echo "🧠 Testing Melvin's Self-Sustaining Learner Features"
echo "=================================================="
echo ""

echo "🎯 Testing confidence-driven output..."
echo "What is consciousness?" | ./melvin
echo ""

echo "📉 Testing confidence decay system..."
echo "confidence decay on" | ./melvin
echo ""

echo "🤔 Testing curiosity-driven learning..."
echo "curiosity on" | ./melvin
echo "What is photosynthesis?" | ./melvin
echo ""

echo "🧠 Testing dual-mode thinking..."
echo "dual thinking on" | ./melvin
echo "How does DNA work?" | ./melvin
echo ""

echo "🔍 Testing meta-reasoning..."
echo "meta reasoning on" | ./melvin
echo "What is creativity?" | ./melvin
echo ""

echo "🎯 Testing adaptive review..."
echo "adaptive review" | ./melvin
echo ""

echo "🔍 Testing knowledge gap analysis..."
echo "check gaps" | ./melvin
echo ""

echo "🎯 Testing evaluation system..."
echo "evaluation on" | ./melvin
echo "evaluate me" | ./melvin
echo ""

echo "✅ Self-sustaining learner features test complete!"
echo ""
echo "Expected behavior:"
echo "- Responses show confidence levels (High/Medium/Low)"
echo "- Confidence decay applied to inactive nodes"
echo "- Self-generated curiosity questions appear"
echo "- Dual-mode thinking shows fast vs deep reasoning"
echo "- Meta-reasoning traces show reflection and confidence"
echo "- Adaptive review prioritizes weak concepts"
echo "- Knowledge gaps trigger repair requests"
echo "- Dynamic evaluation provides real assessment scores"
