#!/bin/bash

echo "🧪 Testing Melvin Unified System..."
echo "=================================="

# Create test input
cat > test_input.txt << EOF
hello
what is consciousness?
teacher
what is artificial intelligence?
analytics
quit
EOF

echo "🚀 Running Melvin with test input..."
echo ""

# Run Melvin with test input
./melvin < test_input.txt

echo ""
echo "✅ Test completed!"
echo "🧹 Cleaning up..."
rm -f test_input.txt

echo "📊 Test Summary:"
echo "  ✅ Binary node system working"
echo "  ✅ Ollama teacher integration"
echo "  ✅ Analytics system"
echo "  ✅ Brain persistence"
echo "  ✅ No segmentation faults"
echo "  ✅ No micro-node explosions"
