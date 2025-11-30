#!/bin/bash

echo "🧪 Testing Melvin's Enhanced Review Cycle with Deep Thinking"
echo "=========================================================="
echo ""

echo "🎯 Testing runtime commands..."
echo "deep think on" | ./melvin
echo ""
echo "review think on" | ./melvin
echo ""

echo "🧠 Testing deep thinking mode with a question..."
echo "What is creativity?" | ./melvin
echo ""

echo "🎯 Disabling deep thinking..."
echo "deep think off" | ./melvin
echo "review think off" | ./melvin
echo ""

echo "✅ Deep thinking system test complete!"
